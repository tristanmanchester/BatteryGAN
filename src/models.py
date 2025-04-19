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

class EfficientSelfAttention(nn.Module):
    """Memory-efficient self-attention mechanism for GANs with downsampling option"""
    def __init__(self, in_dim, downsample_factor=1):
        super().__init__()
        self.downsample_factor = downsample_factor
        self.query_conv = nn.Conv2d(in_dim, in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        # Add downsampling for attention computation if requested
        if downsample_factor > 1:
            self.down = nn.AvgPool2d(kernel_size=downsample_factor, stride=downsample_factor)
            self.up = nn.Upsample(scale_factor=downsample_factor, mode='nearest')
        
    def forward(self, x):
        batch_size, C, height, width = x.size()
        
        # Original queries - maintain full resolution for output
        query = self.query_conv(x)
        
        # Apply downsampling for keys and values to save memory
        if self.downsample_factor > 1:
            x_down = self.down(x)
            key = self.key_conv(x_down)
            value = self.value_conv(x_down)
            ds_height, ds_width = key.shape[2], key.shape[3]
            
            # Reshape and compute attention
            query_flat = query.view(batch_size, -1, height*width).permute(0, 2, 1)
            key_flat = key.view(batch_size, -1, ds_height*ds_width)
            
            # Compute attention with smaller key size
            energy = torch.bmm(query_flat, key_flat)
            # Scale dot-product attention by the dimensionality
            scale_factor = float(key.shape[1]) ** -0.5  
            attention = torch.softmax(energy * scale_factor, dim=2)
            
            # Apply attention to values
            value_flat = value.view(batch_size, -1, ds_height*ds_width)
            out_flat = torch.bmm(attention, value_flat.permute(0, 2, 1))
            
            # Reshape back
            out = out_flat.permute(0, 2, 1).view(batch_size, C, height, width)
            
        else:
            # Traditional self-attention without downsampling
            key = self.key_conv(x)
            value = self.value_conv(x)
            
            # Reshape and compute attention
            query_flat = query.view(batch_size, -1, height*width).permute(0, 2, 1)
            key_flat = key.view(batch_size, -1, height*width)
            
            # Compute attention
            energy = torch.bmm(query_flat, key_flat)
            # Scale dot-product attention by the dimensionality
            scale_factor = float(key.shape[1]) ** -0.5
            attention = torch.softmax(energy * scale_factor, dim=2)
            
            # Apply attention to values
            value_flat = value.view(batch_size, -1, height*width)
            out_flat = torch.bmm(attention, value_flat.permute(0, 2, 1))
            
            # Reshape back
            out = out_flat.permute(0, 2, 1).view(batch_size, C, height, width)
        
        # Residual connection with learnable weight
        return self.gamma * out + x

# Alias for backward compatibility
SelfAttention = EfficientSelfAttention

class Generator(nn.Module):
    """Enhanced generator with residual blocks and self-attention, with optional gradient checkpointing"""
    def __init__(self, in_channels=1, out_channels=1, features=64, n_residual_blocks=9, 
                 self_attention_downsample=1, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        
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
        
        # Individual residual blocks for gradient checkpointing
        self.res_blocks = nn.ModuleList([ResidualBlock(features * 4) for _ in range(n_residual_blocks)])
        
        # Self-attention layer at bottleneck after residual blocks
        self.attention = EfficientSelfAttention(features * 4, downsample_factor=self_attention_downsample)
        
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
    
    def _checkpoint_forward(self, module, x):
        """Forward pass with gradient checkpointing to save memory"""
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(module, x)
        else:
            return module(x)
    
    def _res_block_forward(self, x):
        """Apply all residual blocks with checkpointing if enabled"""
        for res_block in self.res_blocks:
            if self.use_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(res_block, x)
            else:
                x = res_block(x)
        return x
        
    def forward(self, x):
        x = self._checkpoint_forward(self.initial, x)
        x = self._checkpoint_forward(self.down1, x)
        x = self._checkpoint_forward(self.down2, x)
        
        # Apply residual blocks with checkpointing
        x = self._res_block_forward(x)
        
        # Apply attention (if using a large model, can also checkpoint this)
        if self.use_checkpoint and self.training:
            x = torch.utils.checkpoint.checkpoint(self.attention, x)
        else:
            x = self.attention(x)
            
        x = self._checkpoint_forward(self.up1, x)
        x = self._checkpoint_forward(self.up2, x)
        return self._checkpoint_forward(self.final, x)

class Discriminator(nn.Module):
    """Enhanced PatchGAN discriminator with spectral normalization and memory optimizations"""
    def __init__(self, in_channels=1, features=[64, 128, 256, 512], 
                 self_attention_downsample=1, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        
        # Break down sequential modules for gradient checkpointing
        self.layers = nn.ModuleList()
        in_c = in_channels
        
        # First layer without normalization
        self.layers.append(nn.Sequential(
            spectral_norm(nn.Conv2d(in_c, features[0], kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        ))
        in_c = features[0]
        
        # Middle layers with spectral normalization and instance norm
        for feature in features[1:]:
            self.layers.append(nn.Sequential(
                spectral_norm(nn.Conv2d(in_c, feature, kernel_size=4, stride=2, padding=1)),
                nn.InstanceNorm2d(feature),
                nn.LeakyReLU(0.2, inplace=True)
            ))
            in_c = feature
        
        # Add memory-efficient self-attention with downsampling
        self.attention = EfficientSelfAttention(in_c, downsample_factor=self_attention_downsample)
        
        # Final layer
        self.final = spectral_norm(nn.Conv2d(in_c, 1, kernel_size=4, stride=1, padding=1))
    
    def _checkpoint_forward(self, module, x):
        """Forward pass with gradient checkpointing to save memory"""
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(module, x)
        else:
            return module(x)
        
    def forward(self, x):
        # Apply layers with gradient checkpointing if enabled
        for layer in self.layers:
            x = self._checkpoint_forward(layer, x)
            
        # Apply attention with checkpointing if needed
        x = self._checkpoint_forward(self.attention, x)
        
        # Apply final layer
        return self.final(x)

class MultiScaleDiscriminator(nn.Module):
    """Multi-scale discriminator with 2 scales and memory optimizations"""
    def __init__(self, in_channels=1, features=[64, 128, 256, 512], 
                 self_attention_downsample=1, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        
        # Scale 1: Original resolution
        self.disc_original = Discriminator(
            in_channels, 
            features,
            self_attention_downsample=self_attention_downsample,
            use_checkpoint=use_checkpoint
        )
        
        # Scale 2: Half resolution
        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.disc_downsampled = Discriminator(
            in_channels, 
            features, 
            self_attention_downsample=self_attention_downsample,
            use_checkpoint=use_checkpoint
        )
    
    def _checkpoint_forward(self, module, x):
        """Forward pass with gradient checkpointing to save memory"""
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(module, x)
        else:
            return module(x)
        
    def forward(self, x):
        # Get output at original scale
        out_original = self.disc_original(x)
        
        # Downsample input and get output at reduced scale
        x_downsampled = self._checkpoint_forward(self.downsample, x)
        out_downsampled = self.disc_downsampled(x_downsampled)
        
        return out_original, out_downsampled