#!/usr/bin/env python
# Run training script for BatteryGAN with improved architecture & training strategies

import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import itertools
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from datetime import datetime
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler  # Using torch.amp instead of torch.cuda.amp
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.datasets import UnpairedDataset
from src.models import Generator, MultiScaleDiscriminator, Discriminator
from src.train import get_device

class VGGPerceptualLoss(nn.Module):
    """VGG-based perceptual loss"""
    def __init__(self, device):
        super().__init__()
        # Move model to the specified device
        self.device = device
        
        # Use the recommended weights parameter instead of deprecated pretrained
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        
        # Move to correct device and set to eval mode
        vgg = vgg.to(device).eval()
        
        # Use layers up to relu3_3 for feature extraction
        self.model = vgg[:16]
        
        # Freeze the model to prevent training
        for param in self.model.parameters():
            param.requires_grad = False
            
        # ImageNet normalization parameters - ensure they're on the same device
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1))
        
        # Feature comparison criterion
        self.criterion = nn.L1Loss()
        
    def forward(self, x, y):
        # Ensure inputs are on the correct device
        x = x.to(self.device)
        y = y.to(self.device)
        
        # VGG expects 3 channel images, but we have grayscale
        # So we repeat the channel 3 times
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)
        
        # Normalize
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        
        # Get features
        x_features = self.model(x)
        y_features = self.model(y)
        
        # Calculate loss
        return self.criterion(x_features, y_features)

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def update_metrics_plot(metrics, output_dir):
    """Create and update metrics plots during training"""
    metrics_dir = os.path.join(output_dir, "metrics")
    
    # Create figure with two subplots (for generator and discriminators)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
    
    epochs = list(range(1, len(metrics["g_loss"]) + 1))
    
    # Generator losses
    ax1.plot(epochs, metrics["g_loss"], 'b-', label="Generator Loss")
    ax1.set_ylabel("Loss Value")
    ax1.set_title("Generator Loss")
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    
    # Discriminator losses
    ax2.plot(epochs, metrics["d_real_loss"], 'r-', label="D_real Loss")
    ax2.plot(epochs, metrics["d_syn_loss"], 'g-', label="D_syn Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss Value")
    ax2.set_title("Discriminator Losses")
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    # Common settings
    for ax in (ax1, ax2):
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Only show integer ticks
        
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join(metrics_dir, f"losses.png"), dpi=150)
    plt.close()
    
    # Create a learning rate plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics["lr_g"], 'b-', label="Generator LR")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir, f"learning_rates.png"), dpi=150)
    plt.close()
    
    # Create a log-scale plot for discriminator losses (helpful for very small values)
    plt.figure(figsize=(10, 6))
    plt.semilogy(epochs, metrics["d_real_loss"], 'r-', label="D_real Loss")
    plt.semilogy(epochs, metrics["d_syn_loss"], 'g-', label="D_syn Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss Value (log scale)")
    plt.title("Discriminator Losses (Log Scale)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir, f"disc_losses_log.png"), dpi=150)
    plt.close()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train BatteryGAN model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Extract configuration parameters
    real_dir = config['data']['real_dir']
    synthetic_dir = config['data']['synthetic_dir']
    output_dir = config['data']['output_dir']
    
    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "samples"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
    
    # Training parameters
    batch_size = int(config['training']['batch_size'])
    epochs = int(config['training']['epochs'])
    lr_g = float(config['training']['lr_g'])
    lr_d = float(config['training']['lr_d'])
    image_size = int(config['training']['image_size'])
    lambda_cycle = float(config['training']['lambda_cycle'])
    lambda_identity = float(config['training']['lambda_identity'])
    lambda_perceptual = float(config['training']['lambda_perceptual'])
    n_residual_blocks = int(config['training']['n_residual_blocks'])
    use_multi_scale = bool(config['training']['use_multi_scale'])
    
    # Get device and configure device-specific settings
    device = get_device()
    
    # Device-aware mixed precision training
    # Use config value if provided, otherwise auto-detect based on device
    if config['training']['use_amp'] is None:
        use_amp = device.type == 'cuda'  # Automatic mixed precision only on CUDA by default
    else:
        use_amp = config['training']['use_amp']
    
    amp_device_type = device.type  # Store the device type for autocast context
    
    print(f"Training configuration:")
    print(f"- Batch size: {batch_size}")
    print(f"- Image size: {image_size}x{image_size}")
    print(f"- Residual blocks: {n_residual_blocks}")
    print(f"- Multi-scale discriminator: {'Enabled' if use_multi_scale else 'Disabled'}")
    print(f"- Mixed precision (AMP): {'Enabled' if use_amp else 'Disabled'}")
    
    # Define transforms for grayscale images
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # Create dataset and dataloader
    dataset = UnpairedDataset(real_dir, synthetic_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of batches: {len(dataloader)}")
    
    # Initialize models
    G = Generator(in_channels=1, out_channels=1, features=64, n_residual_blocks=n_residual_blocks).to(device)  # Synthetic to Real
    F = Generator(in_channels=1, out_channels=1, features=64, n_residual_blocks=n_residual_blocks).to(device)  # Real to Synthetic
    
    # Use either standard discriminator or multi-scale discriminator
    if use_multi_scale:
        D_real = MultiScaleDiscriminator(in_channels=1).to(device)
        D_syn = MultiScaleDiscriminator(in_channels=1).to(device)
    else:
        D_real = Discriminator(in_channels=1).to(device)
        D_syn = Discriminator(in_channels=1).to(device)
    
    print("Models initialized")
    
    # Initialize VGG perceptual loss
    perceptual_loss = VGGPerceptualLoss(device)
    
    # Define optimizers with different learning rates
    betas = tuple(float(beta) for beta in config['optimizer']['betas'])
    optimizer_G = torch.optim.Adam(
        itertools.chain(G.parameters(), F.parameters()), 
        lr=lr_g, 
        betas=betas
    )
    optimizer_D_real = torch.optim.Adam(D_real.parameters(), lr=lr_d, betas=betas)
    optimizer_D_syn = torch.optim.Adam(D_syn.parameters(), lr=lr_d, betas=betas)
    
    # Learning rate schedulers with minimum LR
    min_lr_factor = config.get('advanced', {}).get('min_lr_factor', 0.2)
    min_lr_g = lr_g * min_lr_factor
    min_lr_d = lr_d * min_lr_factor
    
    scheduler_G = CosineAnnealingLR(optimizer_G, T_max=epochs, eta_min=min_lr_g)
    scheduler_D_real = CosineAnnealingLR(optimizer_D_real, T_max=epochs, eta_min=min_lr_d)
    scheduler_D_syn = CosineAnnealingLR(optimizer_D_syn, T_max=epochs, eta_min=min_lr_d)
    
    # Loss functions
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler() if use_amp else None
    
    # Initialize metrics tracking
    metrics = {
        "g_loss": [],
        "d_real_loss": [],
        "d_syn_loss": [],
        "lr_g": []
    }
    
    print("Starting training...")
    
    for epoch in range(1, epochs + 1):
        total_g_loss = 0
        total_d_real_loss = 0
        total_d_syn_loss = 0
        
        # For tracking batch-level metrics for more detailed analysis
        batch_metrics = {
            "g_loss": [],
            "d_real_loss": [],
            "d_syn_loss": []
        }
        
        for i, batch in enumerate(dataloader):
            real = batch['real'].to(device)
            synthetic = batch['synthetic'].to(device)
            
            # ------------------
            #  Train Generators
            # ------------------
            with autocast(amp_device_type, enabled=use_amp):
                # Get noise standard deviation from config
                noise_std = config.get('advanced', {}).get('noise_std', 0.01)
                
                # Add noise to inputs to prevent discriminator from becoming too confident
                if noise_std > 0:
                    real_noisy = real + torch.randn_like(real) * noise_std
                    synthetic_noisy = synthetic + torch.randn_like(synthetic) * noise_std
                else:
                    real_noisy = real
                    synthetic_noisy = synthetic
                
                # Forward pass
                fake_real = G(synthetic_noisy)  # G(synthetic) = fake_real
                fake_syn = F(real_noisy)        # F(real) = fake_syn
                
                # Cycle consistency
                rec_syn = F(fake_real)    # F(G(synthetic)) = rec_syn
                rec_real = G(fake_syn)    # G(F(real)) = rec_real
                
                # Identity loss (optional but helps preserve characteristics)
                id_real = G(real)         # G(real) should = real
                id_syn = F(synthetic)      # F(synthetic) should = synthetic
                
                # GAN loss
                if use_multi_scale:
                    # For multi-scale discriminator, we get outputs at two scales
                    pred_fake_real_orig, pred_fake_real_down = D_real(fake_real)
                    pred_fake_syn_orig, pred_fake_syn_down = D_syn(fake_syn)
                    
                    valid_real_orig = torch.ones_like(pred_fake_real_orig, device=device)
                    valid_real_down = torch.ones_like(pred_fake_real_down, device=device)
                    valid_syn_orig = torch.ones_like(pred_fake_syn_orig, device=device)
                    valid_syn_down = torch.ones_like(pred_fake_syn_down, device=device)
                    
                    loss_GAN = (criterion_GAN(pred_fake_real_orig, valid_real_orig) + 
                                criterion_GAN(pred_fake_real_down, valid_real_down) +
                                criterion_GAN(pred_fake_syn_orig, valid_syn_orig) +
                                criterion_GAN(pred_fake_syn_down, valid_syn_down)) / 4.0
                else:
                    # Standard discriminator with a single output
                    pred_fake_real = D_real(fake_real)
                    pred_fake_syn = D_syn(fake_syn)
                    valid_real = torch.ones_like(pred_fake_real, device=device)
                    valid_syn = torch.ones_like(pred_fake_syn, device=device)
                    loss_GAN = criterion_GAN(pred_fake_real, valid_real) + criterion_GAN(pred_fake_syn, valid_syn)
                
                # Cycle loss
                loss_cycle = criterion_cycle(rec_syn, synthetic) + criterion_cycle(rec_real, real)
                
                # Identity loss
                loss_identity = criterion_identity(id_real, real) + criterion_identity(id_syn, synthetic)
                
                # Perceptual loss
                loss_perceptual = perceptual_loss(fake_real, synthetic) + perceptual_loss(fake_syn, real)
                
                # Combine all losses
                loss_G = loss_GAN + lambda_cycle * loss_cycle + lambda_identity * loss_identity + lambda_perceptual * loss_perceptual
            
            # Optimize generators
            optimizer_G.zero_grad()
            if use_amp:
                scaler.scale(loss_G).backward()
                scaler.step(optimizer_G)
            else:
                loss_G.backward()
                optimizer_G.step()
            
            # -----------------------
            #  Train Discriminator Real
            # -----------------------
            with autocast(amp_device_type, enabled=use_amp):
                # Instance noise for discriminator (helps with training stability)
                noise_std = config.get('advanced', {}).get('noise_std', 0.01)
                
                # Add noise to inputs to prevent discriminator from becoming too confident
                if noise_std > 0:
                    real_d = real + torch.randn_like(real) * noise_std 
                    fake_real_d = fake_real.detach() + torch.randn_like(fake_real.detach()) * noise_std
                else:
                    real_d = real
                    fake_real_d = fake_real.detach()
                
                if use_multi_scale:
                    # For multi-scale discriminator
                    pred_real_orig, pred_real_down = D_real(real_d)
                    pred_fake_orig, pred_fake_down = D_real(fake_real_d)
                    
                    # Use soft labels (0.9 instead of 1.0, 0.1 instead of 0.0) to prevent overconfidence
                    valid_orig = torch.ones_like(pred_real_orig, device=device) * 0.9
                    valid_down = torch.ones_like(pred_real_down, device=device) * 0.9
                    fake_orig = torch.zeros_like(pred_fake_orig, device=device) + 0.1
                    fake_down = torch.zeros_like(pred_fake_down, device=device) + 0.1
                    
                    real_loss = criterion_GAN(pred_real_orig, valid_orig) + criterion_GAN(pred_real_down, valid_down)
                    fake_loss = criterion_GAN(pred_fake_orig, fake_orig) + criterion_GAN(pred_fake_down, fake_down)
                    loss_D_real = (real_loss + fake_loss) / 4.0
                else:
                    # Standard discriminator
                    pred_real = D_real(real_d)
                    pred_fake = D_real(fake_real_d)
                    
                    # Use soft labels (0.9 instead of 1.0, 0.1 instead of 0.0) to prevent overconfidence
                    valid = torch.ones_like(pred_real, device=device) * 0.9
                    fake = torch.zeros_like(pred_fake, device=device) + 0.1
                    loss_D_real = 0.5 * (criterion_GAN(pred_real, valid) + criterion_GAN(pred_fake, fake))
            
            # Optimize discriminator real
            optimizer_D_real.zero_grad()
            if use_amp:
                scaler.scale(loss_D_real).backward()
                scaler.step(optimizer_D_real)
            else:
                loss_D_real.backward()
                optimizer_D_real.step()
            
            # ---------------------------
            #  Train Discriminator Synthetic
            # ---------------------------
            with autocast(amp_device_type, enabled=use_amp):
                # Apply noise to synthetic discriminator inputs as well
                if noise_std > 0:
                    synthetic_d = synthetic + torch.randn_like(synthetic) * noise_std
                    fake_syn_d = fake_syn.detach() + torch.randn_like(fake_syn.detach()) * noise_std
                else:
                    synthetic_d = synthetic
                    fake_syn_d = fake_syn.detach()
                
                if use_multi_scale:
                    # For multi-scale discriminator
                    pred_real_syn_orig, pred_real_syn_down = D_syn(synthetic_d)
                    pred_fake_syn_orig, pred_fake_syn_down = D_syn(fake_syn_d)
                    
                    # Soft labels for synthetic discriminator too
                    valid_orig = torch.ones_like(pred_real_syn_orig, device=device) * 0.9
                    valid_down = torch.ones_like(pred_real_syn_down, device=device) * 0.9
                    fake_orig = torch.zeros_like(pred_fake_syn_orig, device=device) + 0.1
                    fake_down = torch.zeros_like(pred_fake_syn_down, device=device) + 0.1
                    
                    real_loss = criterion_GAN(pred_real_syn_orig, valid_orig) + criterion_GAN(pred_real_syn_down, valid_down)
                    fake_loss = criterion_GAN(pred_fake_syn_orig, fake_orig) + criterion_GAN(pred_fake_syn_down, fake_down)
                    loss_D_syn = (real_loss + fake_loss) / 4.0
                else:
                    # Standard discriminator
                    pred_real_syn = D_syn(synthetic_d)
                    pred_fake_syn = D_syn(fake_syn_d)
                    valid = torch.ones_like(pred_real_syn, device=device) * 0.9
                    fake = torch.zeros_like(pred_fake_syn, device=device) + 0.1
                    loss_D_syn = 0.5 * (criterion_GAN(pred_real_syn, valid) + criterion_GAN(pred_fake_syn, fake))
            
            # Optimize discriminator synthetic
            optimizer_D_syn.zero_grad()
            if use_amp:
                scaler.scale(loss_D_syn).backward()
                scaler.step(optimizer_D_syn)
            else:
                loss_D_syn.backward()
                optimizer_D_syn.step()
            
            # Update gradient scaler for mixed precision
            if use_amp:
                scaler.update()
            
            # Track losses
            g_loss_val = loss_G.item()
            d_real_loss_val = loss_D_real.item()
            d_syn_loss_val = loss_D_syn.item()
            
            total_g_loss += g_loss_val
            total_d_real_loss += d_real_loss_val
            total_d_syn_loss += d_syn_loss_val
            
            # Store batch-level metrics
            batch_metrics["g_loss"].append(g_loss_val)
            batch_metrics["d_real_loss"].append(d_real_loss_val)
            batch_metrics["d_syn_loss"].append(d_syn_loss_val)
            
            if i % 10 == 0:
                print(f"Epoch [{epoch}/{epochs}] Batch [{i}/{len(dataloader)}] "
                      f"Loss_G: {loss_G.item():.4f} "
                      f"Loss_D_real: {loss_D_real.item():.4f} "
                      f"Loss_D_syn: {loss_D_syn.item():.4f}")
        
        # Update learning rates
        scheduler_G.step()
        scheduler_D_real.step()
        scheduler_D_syn.step()
        
        # Print epoch stats
        avg_g_loss = total_g_loss / len(dataloader)
        avg_d_real_loss = total_d_real_loss / len(dataloader)
        avg_d_syn_loss = total_d_syn_loss / len(dataloader)
        
        current_lr = scheduler_G.get_last_lr()[0]
        print(f"Epoch [{epoch}/{epochs}] "
              f"Avg_Loss_G: {avg_g_loss:.4f} "
              f"Avg_Loss_D_real: {avg_d_real_loss:.4f} "
              f"Avg_Loss_D_syn: {avg_d_syn_loss:.4f} "
              f"LR_G: {current_lr:.6f}")
              
        # Store metrics for plotting
        metrics["g_loss"].append(avg_g_loss)
        metrics["d_real_loss"].append(avg_d_real_loss)
        metrics["d_syn_loss"].append(avg_d_syn_loss)
        metrics["lr_g"].append(current_lr)
        
        # Update metrics plot every few epochs
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            update_metrics_plot(metrics, output_dir)
            
            # Save raw metrics data as CSV for further analysis
            metrics_dir = os.path.join(output_dir, "metrics")
            with open(os.path.join(metrics_dir, "training_metrics.csv"), "w") as f:
                f.write("epoch,g_loss,d_real_loss,d_syn_loss,lr_g\n")
                for i in range(len(metrics["g_loss"])):
                    f.write(f"{i+1},{metrics['g_loss'][i]},{metrics['d_real_loss'][i]},{metrics['d_syn_loss'][i]},{metrics['lr_g'][i]}\n")
            
            # Save batch-level metrics for the current epoch
            with open(os.path.join(metrics_dir, f"batch_metrics_epoch{epoch}.csv"), "w") as f:
                f.write("batch,g_loss,d_real_loss,d_syn_loss\n")
                for i in range(len(batch_metrics["g_loss"])):
                    f.write(f"{i+1},{batch_metrics['g_loss'][i]},{batch_metrics['d_real_loss'][i]},{batch_metrics['d_syn_loss'][i]}\n")
        
            # Plot batch-level metrics for the current epoch
            plt.figure(figsize=(12, 8))
            batches = list(range(1, len(batch_metrics["g_loss"]) + 1))
            plt.plot(batches, batch_metrics["g_loss"], 'b-', label="Generator Loss", alpha=0.7)
            plt.plot(batches, batch_metrics["d_real_loss"], 'r-', label="D_real Loss", alpha=0.7)
            plt.plot(batches, batch_metrics["d_syn_loss"], 'g-', label="D_syn Loss", alpha=0.7)
            plt.xlabel("Batch")
            plt.ylabel("Loss Value")
            plt.title(f"Batch-level Losses for Epoch {epoch}")
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(metrics_dir, f"batch_losses_epoch{epoch}.png"), dpi=150)
            plt.close()
        
        # Save sample outputs based on config
        save_sample_every = config['visualization']['save_sample_every']
        if epoch % save_sample_every == 0 or epoch == 1:
            from src.utils import save_image
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Get the first image from each batch
            sample_real = real[0]
            sample_syn = synthetic[0]
            sample_fake_real = fake_real[0]
            sample_fake_syn = fake_syn[0]
            
            # Create directory for visualizations
            viz_dir = os.path.join(output_dir, "visualizations")
            os.makedirs(viz_dir, exist_ok=True)
            
            # Function to save normalized images with proper visualization
            def save_tensor_image(tensor, path):
                # Ensure tensor is on CPU and detached from graph
                img = tensor.detach().cpu()
                
                # Denormalize the tensor (from [-1,1] to [0,1])
                img = img.mul(0.5).add(0.5)
                
                # Convert to numpy for visualization
                if img.dim() == 3 and img.size(0) == 1:  # Single channel
                    img_np = img.squeeze(0).numpy()  # Convert to [H,W] for grayscale display
                else:  # RGB
                    img_np = img.permute(1, 2, 0).numpy()  # Convert to [H,W,C] for RGB display
                
                # Create a visualization with matplotlib
                plt.figure(figsize=(8, 8))
                plt.imshow(img_np, cmap='gray', vmin=0, vmax=1)
                plt.colorbar()
                plt.axis('off')
                plt.title(os.path.basename(path).split('.')[0])
                plt.tight_layout()
                plt.savefig(path)
                plt.close()
                
                # Also save the raw image through our standard pipeline
                # Make sure to denormalize the tensor from [-1,1] to [0,1] before saving
                normalized_tensor = tensor.detach().cpu().mul(0.5).add(0.5) 
                save_image(normalized_tensor, path.replace('visualizations', 'samples'))
            
            # Save images with proper visualization
            save_tensor_image(sample_real, os.path.join(viz_dir, f"real_epoch{epoch}.png"))
            save_tensor_image(sample_syn, os.path.join(viz_dir, f"synthetic_epoch{epoch}.png"))
            save_tensor_image(sample_fake_real, os.path.join(viz_dir, f"fake_real_epoch{epoch}.png"))
            save_tensor_image(sample_fake_syn, os.path.join(viz_dir, f"fake_syn_epoch{epoch}.png"))
            
            # Create a compact side-by-side comparison
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            fig.subplots_adjust(wspace=0.05, hspace=0.2)  # Reduce space between plots
            
            # Convert all tensors to numpy for display
            real_np = sample_real.detach().cpu().mul(0.5).add(0.5).squeeze(0).numpy()
            syn_np = sample_syn.detach().cpu().mul(0.5).add(0.5).squeeze(0).numpy()
            fake_real_np = sample_fake_real.detach().cpu().mul(0.5).add(0.5).squeeze(0).numpy()
            fake_syn_np = sample_fake_syn.detach().cpu().mul(0.5).add(0.5).squeeze(0).numpy()
            
            # Plot all images with proper titles and without axes
            axes[0, 0].imshow(real_np, cmap='gray', vmin=0, vmax=1)
            axes[0, 0].set_title("Real Image", fontsize=12)
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(syn_np, cmap='gray', vmin=0, vmax=1)
            axes[0, 1].set_title("Synthetic Image", fontsize=12)
            axes[0, 1].axis('off')
            
            axes[1, 0].imshow(fake_real_np, cmap='gray', vmin=0, vmax=1)
            axes[1, 0].set_title("Fake Real (Syn→Real)", fontsize=12)
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(fake_syn_np, cmap='gray', vmin=0, vmax=1)
            axes[1, 1].set_title("Fake Synthetic (Real→Syn)", fontsize=12)
            axes[1, 1].axis('off')
            
            # Add epoch information to each title instead of using a suptitle
            for ax in axes.flatten():
                title = ax.get_title()
                ax.set_title(f"{title}\nEpoch {epoch}/{epochs}", fontsize=12)
            
            # Ensure the layout is tight without reserving space for suptitle
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, f"comparison_epoch{epoch}.png"), dpi=150, bbox_inches='tight')
            plt.close()
            
            # Also create a 2×2 grid of all samples with a shared colorbar for visual analysis
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            fig.subplots_adjust(wspace=0.05, hspace=0.05, right=0.85)  # Make room for colorbar
            
            # Common colormap and normalization for all plots
            vmin, vmax = 0, 1
            imgs = []
            imgs.append(axes[0, 0].imshow(real_np, cmap='gray', vmin=vmin, vmax=vmax))
            axes[0, 0].set_title("Real Image", fontsize=12)
            axes[0, 0].axis('off')
            
            imgs.append(axes[0, 1].imshow(syn_np, cmap='gray', vmin=vmin, vmax=vmax))
            axes[0, 1].set_title("Synthetic Image", fontsize=12)
            axes[0, 1].axis('off')
            
            imgs.append(axes[1, 0].imshow(fake_real_np, cmap='gray', vmin=vmin, vmax=vmax))
            axes[1, 0].set_title("Fake Real (Syn→Real)", fontsize=12)
            axes[1, 0].axis('off')
            
            imgs.append(axes[1, 1].imshow(fake_syn_np, cmap='gray', vmin=vmin, vmax=vmax))
            axes[1, 1].set_title("Fake Synthetic (Real→Syn)", fontsize=12)
            axes[1, 1].axis('off')
            
            # Add a shared colorbar
            cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
            cb = fig.colorbar(imgs[0], cax=cbar_ax)
            cb.set_label('Pixel Intensity')
            
            # Add epoch information to each title instead of using a suptitle
            for ax in axes.flatten():
                title = ax.get_title()
                ax.set_title(f"{title}\nEpoch {epoch}", fontsize=12)
                
            # Ensure the layout is tight without reserving space for suptitle
            fig.tight_layout(rect=[0, 0, 0.85, 1.0])  # Only account for colorbar space
            plt.savefig(os.path.join(viz_dir, f"analysis_epoch{epoch}.png"), dpi=150, bbox_inches='tight')
            plt.close()
        
        # Save models based on config
        save_model_every = config['visualization']['save_model_every']
        if epoch % save_model_every == 0 or epoch == epochs:
            models_dir = os.path.join(output_dir, "models")
            torch.save(G.state_dict(), os.path.join(models_dir, f'G_epoch{epoch}.pth'))
            torch.save(F.state_dict(), os.path.join(models_dir, f'F_epoch{epoch}.pth'))
            torch.save(D_real.state_dict(), os.path.join(models_dir, f'D_real_epoch{epoch}.pth'))
            torch.save(D_syn.state_dict(), os.path.join(models_dir, f'D_syn_epoch{epoch}.pth'))
    
    print('Training complete.')

if __name__ == '__main__':
    main()