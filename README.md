# BatteryGAN

A CycleGAN-based data generator for tomography data with enhanced architecture. This project takes real tomography data and synthetically generated data and labels, then trains an advanced CycleGAN to produce realistic data while preserving label information.

**Key Features:**
- Trains on unpaired real and synthetic tomography data
- Enhanced architecture with residual blocks and self-attention
- Multi-scale discriminator for improved quality
- Perceptual and identity losses for better structure preservation
- Supports grayscale (1-channel) TIFF images
- Cross-platform: macOS (Apple MPS), Linux/Windows (CUDA) with automatic hardware detection
- Mixed precision training on compatible devices

**Data Structure:**
- Real data: `data/real_data/`
- Synthetic data: `data/synthetic_data/data/`
- Synthetic labels: `data/synthetic_data/labels/`

## Project Requirements

- Python >= 3.8
- macOS with Apple Metal GPU (MPS) or Linux/Windows with CUDA-capable GPU
- `pixi` environment manager (initialized)

## Dependencies

Install dependencies with pixi (see `pixi.toml` for full list):

```bash
pixi install
```

PyTorch is installed via PyPI for Apple MPS or CUDA support.

## Project Structure

```bash
BatteryGAN/
├── data/
│   ├── real_data/                   # Real tomography data
│   └── synthetic_data/
│       ├── data/                    # Synthetic data
│       └── labels/                  # Synthetic labels
├── src/
│   ├── __init__.py
│   ├── datasets.py                  # Unpaired dataset loader
│   ├── models.py                    # Enhanced CycleGAN architecture
│   ├── train.py                     # Training logic with device handling
│   ├── utils.py                     # TIFF/grayscale utilities
│   └── label_dataset.py             # Paired dataset for synthetic+labels
├── run_training.py                  # Main training script with advanced losses
├── generate.py                      # Generate realistic data from synthetic
├── output/                          # Model checkpoints and samples
├── generated_data/                  # Output from generate.py
├── notebooks/                       # Jupyter notebooks
│   └── .gitkeep
├── README.md
├── CLAUDE.md                        # Guidelines for Claude Code
└── .gitignore
```

## Usage

### 1. Train the Enhanced CycleGAN

```bash
pixi run python run_training.py --config config.yaml
```
- Configuration is loaded from the YAML config file
- Automatically selects the best available device (CUDA, MPS, or CPU)
- Features 9 residual blocks, multi-scale discriminator, and perceptual loss
- Model checkpoints and sample images will be saved to `output/`

You can create custom configuration files and specify them:
```bash
pixi run python run_training.py --config custom_config.yaml
```

### 2. Generate Realistic Data

After training, generate new data with:

```bash
pixi run python generate.py \
  --model_path output/G_epoch50.pth \
  --synthetic_dir data/synthetic_data/data \
  --labels_dir data/synthetic_data/labels \
  --output_dir generated_data \
  --image_size 512 \
  --residual_blocks 9
```
- This will create `generated_data/data/` (realistic images) and `generated_data/labels/` (copied labels) 
- The `--residual_blocks` parameter should match what was used during training
- All images will be resized to the specified `--image_size` during processing

## Advanced Configuration

### Config File Structure

Training parameters can be configured using a YAML file:

```yaml
# Data paths
data:
  real_dir: "./data/real_data"
  synthetic_dir: "./data/synthetic_data/data"
  output_dir: "./output"

# Training parameters
training:
  batch_size: 1
  epochs: 50
  lr_g: 1e-4  # Generator learning rate
  lr_d: 4e-4  # Discriminator learning rate (TTUR)
  image_size: 256  # Image size
  lambda_cycle: 10.0  # Cycle consistency weight
  lambda_identity: 5.0  # Identity loss weight
  lambda_perceptual: 1.0  # Perceptual loss weight
  n_residual_blocks: 9  # Number of residual blocks in generator
  use_multi_scale: true  # Whether to use multi-scale discriminator
  use_amp: null  # Set to null to auto-detect (true for CUDA, false for MPS)

# Optimizer settings
optimizer:
  betas: [0.5, 0.999]  # Adam optimizer betas

# Visualization settings
visualization:
  save_sample_every: 5  # Save sample images every N epochs
  save_model_every: 10  # Save model checkpoints every N epochs
```

See `config.yaml` for default values.

### Model Improvements

The enhanced model includes several improvements:
- **Residual Blocks**: Deeper network with 9 residual blocks for better feature extraction
- **Self-Attention**: Captures long-range dependencies in the image
- **Multi-Scale Discriminator**: Evaluates generated images at multiple scales
- **Perceptual Loss**: Uses VGG features to maintain structural consistency
- **Identity Loss**: Preserves domain-specific characteristics
- **Two-Timescale Update Rule (TTUR)**: Different learning rates for generators vs. discriminators
- **Learning Rate Scheduling**: Cosine annealing for gradual reduction
- **Mixed Precision Training**: Automatic FP16 on compatible CUDA devices

## Notes
- All images are handled as grayscale (1-channel) TIFFs
- Automatic device detection for cross-platform compatibility
- Training and generation scripts are configurable via command-line arguments

## Next Steps
- Visualize results in Jupyter notebooks
- Evaluate generated data quality
- Fine-tune model or add new features as needed