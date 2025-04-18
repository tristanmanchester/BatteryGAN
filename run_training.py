#!/usr/bin/env python
# Run training script for BatteryGAN with improved architecture & training strategies

import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import itertools
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
        
        try:
            # Try to load with pretrained weights
            vgg = models.vgg19(pretrained=True).features
        except Exception as e:
            # Fallback for newer PyTorch versions
            print(f"Warning: {e}. Trying with weights parameter instead...")
            vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        
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

def main():
    # Define paths
    real_dir = "./data/real_data"
    synthetic_dir = "./data/synthetic_data/data"
    output_dir = "./output"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "samples"), exist_ok=True)
    
    # Training parameters
    batch_size = 1
    epochs = 50
    lr_g = 1e-4  # Generator learning rate
    lr_d = 4e-4  # Discriminator learning rate (TTUR)
    image_size = 256  # Image size
    lambda_cycle = 10.0  # Cycle consistency weight
    lambda_identity = 5.0  # Identity loss weight
    lambda_perceptual = 1.0  # Perceptual loss weight
    n_residual_blocks = 9  # Number of residual blocks in generator
    use_multi_scale = True  # Whether to use multi-scale discriminator
    
    # Get device and configure device-specific settings
    device = get_device()
    
    # Device-aware mixed precision training
    # Only enable AMP on CUDA by default; MPS has limited mixed precision support
    use_amp = device.type == 'cuda'  # Automatic mixed precision only on CUDA
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
    optimizer_G = torch.optim.Adam(
        itertools.chain(G.parameters(), F.parameters()), 
        lr=lr_g, 
        betas=(0.5, 0.999)
    )
    optimizer_D_real = torch.optim.Adam(D_real.parameters(), lr=lr_d, betas=(0.5, 0.999))
    optimizer_D_syn = torch.optim.Adam(D_syn.parameters(), lr=lr_d, betas=(0.5, 0.999))
    
    # Learning rate schedulers
    scheduler_G = CosineAnnealingLR(optimizer_G, T_max=epochs)
    scheduler_D_real = CosineAnnealingLR(optimizer_D_real, T_max=epochs)
    scheduler_D_syn = CosineAnnealingLR(optimizer_D_syn, T_max=epochs)
    
    # Loss functions
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler() if use_amp else None
    
    print("Starting training...")
    
    for epoch in range(1, epochs + 1):
        total_g_loss = 0
        total_d_real_loss = 0
        total_d_syn_loss = 0
        
        for i, batch in enumerate(dataloader):
            real = batch['real'].to(device)
            synthetic = batch['synthetic'].to(device)
            
            # ------------------
            #  Train Generators
            # ------------------
            with autocast(amp_device_type, enabled=use_amp):
                # Forward pass
                fake_real = G(synthetic)  # G(synthetic) = fake_real
                fake_syn = F(real)        # F(real) = fake_syn
                
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
                if use_multi_scale:
                    # For multi-scale discriminator
                    pred_real_orig, pred_real_down = D_real(real)
                    pred_fake_orig, pred_fake_down = D_real(fake_real.detach())
                    
                    valid_orig = torch.ones_like(pred_real_orig, device=device)
                    valid_down = torch.ones_like(pred_real_down, device=device)
                    fake_orig = torch.zeros_like(pred_fake_orig, device=device)
                    fake_down = torch.zeros_like(pred_fake_down, device=device)
                    
                    real_loss = criterion_GAN(pred_real_orig, valid_orig) + criterion_GAN(pred_real_down, valid_down)
                    fake_loss = criterion_GAN(pred_fake_orig, fake_orig) + criterion_GAN(pred_fake_down, fake_down)
                    loss_D_real = (real_loss + fake_loss) / 4.0
                else:
                    # Standard discriminator
                    pred_real = D_real(real)
                    pred_fake = D_real(fake_real.detach())
                    valid = torch.ones_like(pred_real, device=device)
                    fake = torch.zeros_like(pred_fake, device=device)
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
                if use_multi_scale:
                    # For multi-scale discriminator
                    pred_real_syn_orig, pred_real_syn_down = D_syn(synthetic)
                    pred_fake_syn_orig, pred_fake_syn_down = D_syn(fake_syn.detach())
                    
                    valid_orig = torch.ones_like(pred_real_syn_orig, device=device)
                    valid_down = torch.ones_like(pred_real_syn_down, device=device)
                    fake_orig = torch.zeros_like(pred_fake_syn_orig, device=device)
                    fake_down = torch.zeros_like(pred_fake_syn_down, device=device)
                    
                    real_loss = criterion_GAN(pred_real_syn_orig, valid_orig) + criterion_GAN(pred_real_syn_down, valid_down)
                    fake_loss = criterion_GAN(pred_fake_syn_orig, fake_orig) + criterion_GAN(pred_fake_syn_down, fake_down)
                    loss_D_syn = (real_loss + fake_loss) / 4.0
                else:
                    # Standard discriminator
                    pred_real_syn = D_syn(synthetic)
                    pred_fake_syn = D_syn(fake_syn.detach())
                    valid = torch.ones_like(pred_real_syn, device=device)
                    fake = torch.zeros_like(pred_fake_syn, device=device)
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
            total_g_loss += loss_G.item()
            total_d_real_loss += loss_D_real.item()
            total_d_syn_loss += loss_D_syn.item()
            
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
        
        print(f"Epoch [{epoch}/{epochs}] "
              f"Avg_Loss_G: {avg_g_loss:.4f} "
              f"Avg_Loss_D_real: {avg_d_real_loss:.4f} "
              f"Avg_Loss_D_syn: {avg_d_syn_loss:.4f} "
              f"LR_G: {scheduler_G.get_last_lr()[0]:.6f}")
        
        # Save sample outputs every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
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
            
            # Add epoch information as a suptitle
            plt.suptitle(f"Epoch {epoch}/{epochs}", fontsize=14)
            
            # Ensure the layout is tight
            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave room for suptitle
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
            
            plt.suptitle(f"Comparison at Epoch {epoch}", fontsize=14)
            plt.savefig(os.path.join(viz_dir, f"analysis_epoch{epoch}.png"), dpi=150, bbox_inches='tight')
            plt.close()
        
        # Save models every 10 epochs
        if epoch % 10 == 0 or epoch == epochs:
            torch.save(G.state_dict(), os.path.join(output_dir, f'G_epoch{epoch}.pth'))
            torch.save(F.state_dict(), os.path.join(output_dir, f'F_epoch{epoch}.pth'))
            torch.save(D_real.state_dict(), os.path.join(output_dir, f'D_real_epoch{epoch}.pth'))
            torch.save(D_syn.state_dict(), os.path.join(output_dir, f'D_syn_epoch{epoch}.pth'))
    
    print('Training complete.')

if __name__ == '__main__':
    main()