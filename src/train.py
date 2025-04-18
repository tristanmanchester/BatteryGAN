import os
import argparse
import itertools
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from src.datasets import UnpairedDataset
from src.models import Generator, Discriminator
from src.utils import save_image

def get_device():
    """
    Get the best available device for PyTorch training.
    
    Returns:
        torch.device: CUDA for NVIDIA GPUs, MPS for Apple Silicon, or CPU as fallback
    """
    # Check for CUDA first (NVIDIA GPUs)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_name = torch.cuda.get_device_name(0)
        print(f"Using CUDA device: {device_name}")
        # Print available VRAM
        free_mem, total_mem = torch.cuda.mem_get_info(0)
        print(f"Available VRAM: {free_mem/1024**3:.2f}GB / {total_mem/1024**3:.2f}GB total")
        return device
    
    # Check for MPS (Apple Silicon M-series chips)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"Using Apple Metal Performance Shaders (MPS) device")
        return device
    
    # Fallback to CPU
    else:
        device = torch.device('cpu')
        print(f"No GPU found. Using CPU. Training will be slow.")
        return device


def main():
    parser = argparse.ArgumentParser(description='Train CycleGAN on tomography data')
    parser.add_argument('--real_dir', type=str, required=True)
    parser.add_argument('--synthetic_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--image_size', type=int, default=256)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    samples_dir = os.path.join(args.output_dir, 'samples')
    os.makedirs(samples_dir, exist_ok=True)

    device = get_device()
    print(f'Using device: {device}')

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = UnpairedDataset(args.real_dir, args.synthetic_dir, transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

    # Initialize models
    G = Generator().to(device)
    F = Generator().to(device)
    D_real = Discriminator().to(device)
    D_syn = Discriminator().to(device)

    # Losses
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    lambda_cycle = 10.0

    # Optimizers
    optimizer_G = torch.optim.Adam(itertools.chain(G.parameters(), F.parameters()), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D_real = torch.optim.Adam(D_real.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D_syn = torch.optim.Adam(D_syn.parameters(), lr=args.lr, betas=(0.5, 0.999))

    for epoch in range(1, args.epochs + 1):
        for i, batch in enumerate(dataloader):
            real = batch['real'].to(device)
            synthetic = batch['synthetic'].to(device)

            # ------------------
            #  Train Generators
            # ------------------
            fake_real = G(synthetic)
            fake_syn = F(real)
            rec_syn = F(fake_real)
            rec_real = G(fake_syn)

            # GAN loss
            pred_fake_real = D_real(fake_real)
            pred_fake_syn = D_syn(fake_syn)
            valid_real = torch.ones_like(pred_fake_real, device=device)
            valid_syn = torch.ones_like(pred_fake_syn, device=device)
            loss_GAN = criterion_GAN(pred_fake_real, valid_real) + criterion_GAN(pred_fake_syn, valid_syn)

            # Cycle loss
            loss_cycle = criterion_cycle(rec_syn, synthetic) + criterion_cycle(rec_real, real)

            loss_G = loss_GAN + lambda_cycle * loss_cycle

            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            # -----------------------
            #  Train Discriminator Real
            # -----------------------
            pred_real = D_real(real)
            pred_fake = D_real(fake_real.detach())
            valid = torch.ones_like(pred_real, device=device)
            fake = torch.zeros_like(pred_fake, device=device)
            loss_D_real = 0.5 * (criterion_GAN(pred_real, valid) + criterion_GAN(pred_fake, fake))

            optimizer_D_real.zero_grad()
            loss_D_real.backward()
            optimizer_D_real.step()

            # ---------------------------
            #  Train Discriminator Synthetic
            # ---------------------------
            pred_real_syn = D_syn(synthetic)
            pred_fake_syn = D_syn(fake_syn.detach())
            valid = torch.ones_like(pred_real_syn, device=device)
            fake = torch.zeros_like(pred_fake_syn, device=device)
            loss_D_syn = 0.5 * (criterion_GAN(pred_real_syn, valid) + criterion_GAN(pred_fake_syn, fake))

            optimizer_D_syn.zero_grad()
            loss_D_syn.backward()
            optimizer_D_syn.step()

            if i % 100 == 0:
                print(f"Epoch [{epoch}/{args.epochs}] Batch [{i}/{len(dataloader)}] "
                      f"Loss_G: {loss_G.item():.4f} "
                      f"Loss_D_real: {loss_D_real.item():.4f} "
                      f"Loss_D_syn: {loss_D_syn.item():.4f}")

        # Save sample outputs and models
        save_image(fake_real.data[0], os.path.join(samples_dir, f"fake_real_epoch{epoch}.png"))
        save_image(fake_syn.data[0], os.path.join(samples_dir, f"fake_syn_epoch{epoch}.png"))

        torch.save(G.state_dict(), os.path.join(args.output_dir, 'G.pth'))
        torch.save(F.state_dict(), os.path.join(args.output_dir, 'F.pth'))
        torch.save(D_real.state_dict(), os.path.join(args.output_dir, 'D_real.pth'))
        torch.save(D_syn.state_dict(), os.path.join(args.output_dir, 'D_syn.pth'))

    print('Training complete.')

if __name__ == '__main__':
    main()
