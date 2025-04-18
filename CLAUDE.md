# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build/Run Commands
- Run training: `pixi run python run_training.py`
- Generate data: `pixi run python generate.py --model_path <path> --synthetic_dir <dir> --labels_dir <dir> --output_dir <dir> --image_size 512 --residual_blocks 9`
- Install dependencies: `pixi install`
- Debug image loading: `pixi run python debug_image.py`

## Code Style Guidelines
- Imports: Use explicit imports (no wildcard imports). Order: stdlib → third-party → local modules
- Formatting: 4-space indentation, max 100 chars per line
- Type hints: Encouraged but not strictly enforced throughout codebase
- Naming: snake_case for functions/variables, CamelCase for classes
- Variables: Descriptive names, esp. for tensors (e.g., real_batch, fake_syn_img)
- PyTorch: Use device-agnostic code with device detection via get_device()
- Error handling: Use explicit error messages, validate paths before file operations
- Batch processing: Maintain batch dimension in tensor operations
- Image handling: Grayscale (1-channel) TIFF images are standard format
- Resource management: Use try/finally blocks with cleanup() to ensure proper resource release

## Important Implementation Details
- Cross-platform: Code must work on both CUDA (NVIDIA) and MPS (Apple Silicon) by using device-aware logic
- Tensor devices: Always use .to(device) for tensors and ensure they match across operations
- Mixed precision: Use torch.amp.autocast(device_type) with device-specific parameters
- Graceful shutdown: Handle SIGINT and SIGTERM signals to clean up resources when interrupted
- Generator: Uses 9 residual blocks and self-attention for enhanced performance
- Loss functions: Multiple components (GAN, cycle, identity, perceptual) weighted with lambda parameters
- Image processing: Always normalize from [-1,1] to [0,1] range when saving output images
- Multi-scale discriminator: Returns outputs at two scales that must be handled correctly