#!/usr/bin/env python
# Generate realistic data from synthetic data using trained CycleGAN

import os
import argparse
import torch
import torchvision.transforms as transforms
from pathlib import Path
from src.models import Generator
from src.utils import load_image, save_image
from src.train import get_device

def main():
    parser = argparse.ArgumentParser(description='Generate realistic data from synthetic data')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the generator model file')
    parser.add_argument('--synthetic_dir', type=str, required=True, help='Directory with synthetic data')
    parser.add_argument('--labels_dir', type=str, default=None, help='Directory with label data (optional)')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for generated data')
    parser.add_argument('--image_size', type=int, default=512, help='Size to resize images to')
    parser.add_argument('--residual_blocks', type=int, default=9, help='Number of residual blocks in generator')
    args = parser.parse_args()

    # Create output directories
    output_data_dir = os.path.join(args.output_dir, 'data')
    os.makedirs(output_data_dir, exist_ok=True)
    
    if args.labels_dir:
        output_labels_dir = os.path.join(args.output_dir, 'labels')
        os.makedirs(output_labels_dir, exist_ok=True)

    # Get device
    device = get_device()

    # Load the generator model with improved architecture
    model = Generator(
        in_channels=1, 
        out_channels=1, 
        features=64, 
        n_residual_blocks=args.residual_blocks
    )
    
    print(f"Loading model from: {args.model_path}")
    # Safely load the model state dict with device mapping
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Successfully loaded model with {args.residual_blocks} residual blocks")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying to load with different model configuration...")
        # If first attempt fails, try with default parameters as fallback
        model = Generator(in_channels=1, out_channels=1)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print("Successfully loaded model with default architecture")
    
    model.to(device)
    model.eval()
    print(f"Loaded enhanced generator model from {args.model_path}")

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # Get list of synthetic images
    synthetic_paths = sorted([os.path.join(args.synthetic_dir, f) for f in os.listdir(args.synthetic_dir) 
                             if f.endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg'))])
    
    # Get list of label images if provided
    label_paths = []
    if args.labels_dir:
        label_paths = sorted([os.path.join(args.labels_dir, f) for f in os.listdir(args.labels_dir)
                             if f.endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg'))])
        
        # Verify that we have the same number of synthetic and label images
        if len(synthetic_paths) != len(label_paths):
            print(f"Warning: Number of synthetic images ({len(synthetic_paths)}) does not match number of label images ({len(label_paths)})")

    print(f"Found {len(synthetic_paths)} synthetic images")
    if args.labels_dir:
        print(f"Found {len(label_paths)} label images")

    # Process each image
    for i, syn_path in enumerate(synthetic_paths):
        print(f"Processing image {i+1}/{len(synthetic_paths)}: {os.path.basename(syn_path)}")
        
        # Load and transform synthetic image
        syn_img = load_image(syn_path, grayscale=True)
        syn_tensor = transform(syn_img).unsqueeze(0).to(device)
        
        # Generate realistic image
        with torch.no_grad():
            fake_real = model(syn_tensor)
        
        # Save generated image (normalizing from [-1,1] to [0,1] range)
        output_name = f"generated_{os.path.basename(syn_path)}"
        if not output_name.endswith('.tif'):
            output_name = output_name.split('.')[0] + '.tif'
        output_path = os.path.join(output_data_dir, output_name)
        # Denormalize from [-1,1] to [0,1] range before saving
        normalized_fake_real = fake_real[0].mul(0.5).add(0.5)
        save_image(normalized_fake_real, output_path, grayscale=True)
        
        # If we have labels, copy them to the output directory
        if args.labels_dir and i < len(label_paths):
            label_path = label_paths[i]
            output_label_name = f"label_{os.path.basename(syn_path)}"
            if not output_label_name.endswith('.tif'):
                output_label_name = output_label_name.split('.')[0] + '.tif'
            output_label_path = os.path.join(output_labels_dir, output_label_name)
            
            # For labels, we just copy them without transformation
            import shutil
            shutil.copy(label_path, output_label_path)
    
    print(f"Generation complete. Generated {len(synthetic_paths)} images in {output_data_dir}")
    if args.labels_dir:
        print(f"Copied {min(len(synthetic_paths), len(label_paths))} label images to {output_labels_dir}")

def cleanup():
    """Clean up resources, especially GPU memory"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            print("\nCUDA memory cleared.")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # MPS doesn't have explicit memory management functions like CUDA
            print("\nNote: MPS (Apple Silicon) doesn't provide explicit memory cleanup functions.")
        print("Resources cleaned up. Shutdown complete.")
    except Exception as e:
        print(f"Warning: Cleanup encountered an error: {e}")

def signal_handler(sig, frame):
    """Handle interrupt signals like Ctrl+C"""
    print(f"\n\nInterrupt signal received ({sig}). Cleaning up and exiting gracefully...")
    cleanup()
    print("Generation halted. Any completed images should be in the output directory.")
    sys.exit(0)

if __name__ == '__main__':
    import signal
    import sys
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler) # Termination request
    
    try:
        main()
    except Exception as e:
        print(f"\nError during generation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Always run cleanup whether generation succeeded or failed
        cleanup()
