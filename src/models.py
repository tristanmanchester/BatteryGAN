import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class ResidualBlock(nn.Module):
    """Residual block with instance normalization"""
    def __init__(self, features):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(features),
            nn.ReLU(True),
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(features)
        )
        
    def forward(self, x):
        return x + self.block(x)

class SelfAttention(nn.Module):
    """Self-attention mechanism for GANs"""
    def __init__(self, in_dim):
        super().__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, C, width, height = x.size()
        query = self.query_conv(x).view(batch_size, -1, width*height).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, width*height)
        energy = torch.bmm(query, key)
        attention = torch.softmax(energy, dim=2)
        value = self.value_conv(x).view(batch_size, -1, width*height)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        return self.gamma * out + x

class Generator(nn.Module):
    """Enhanced generator with residual blocks and self-attention"""
    def __init__(self, in_channels=1, out_channels=1, features=64, n_residual_blocks=9):
        super().__init__()
        # Initial convolution
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, features, kernel_size=7, stride=1),
            nn.InstanceNorm2d(features),
            nn.ReLU(True)
        )
        
        # Downsampling
        self.down1 = nn.Sequential(
            nn.Conv2d(features, features * 2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(features * 2),
            nn.ReLU(True)
        )
        
        self.down2 = nn.Sequential(
            nn.Conv2d(features * 2, features * 4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(features * 4),
            nn.ReLU(True)
        )
        
        # Residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(features * 4) for _ in range(n_residual_blocks)]
        )
        
        # Self-attention layer at bottleneck after residual blocks
        self.attention = SelfAttention(features * 4)
        
        # Upsampling
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(features * 4, features * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(features * 2),
            nn.ReLU(True)
        )
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(features * 2, features, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(features),
            nn.ReLU(True)
        )
        
        # Final convolution
        self.final = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(features, out_channels, kernel_size=7, stride=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.initial(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.residual_blocks(x)
        x = self.attention(x)
        x = self.up1(x)
        x = self.up2(x)
        return self.final(x)

class Discriminator(nn.Module):
    """Enhanced PatchGAN discriminator with spectral normalization"""
    def __init__(self, in_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        layers = []
        in_c = in_channels
        
        # First layer without normalization
        layers.append(spectral_norm(nn.Conv2d(in_c, features[0], kernel_size=4, stride=2, padding=1)))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        in_c = features[0]
        
        # Middle layers with spectral normalization and instance norm
        for feature in features[1:]:
            layers.append(spectral_norm(nn.Conv2d(in_c, feature, kernel_size=4, stride=2, padding=1)))
            layers.append(nn.InstanceNorm2d(feature))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_c = feature
            
        # Add self-attention after the second to last layer
        self.main = nn.Sequential(*layers)
        self.attention = SelfAttention(in_c)
        
        # Final layer
        self.final = spectral_norm(nn.Conv2d(in_c, 1, kernel_size=4, stride=1, padding=1))
        
    def forward(self, x):
        x = self.main(x)
        x = self.attention(x)
        return self.final(x)

class MultiScaleDiscriminator(nn.Module):
    """Multi-scale discriminator with 2 scales"""
    def __init__(self, in_channels=1):
        super().__init__()
        # Scale 1: Original resolution
        self.disc_original = Discriminator(in_channels)
        
        # Scale 2: Half resolution
        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.disc_downsampled = Discriminator(in_channels)
        
    def forward(self, x):
        # Get output at original scale
        out_original = self.disc_original(x)
        
        # Downsample input and get output at reduced scale
        x_downsampled = self.downsample(x)
        out_downsampled = self.disc_downsampled(x_downsampled)
        
        return out_original, out_downsampled