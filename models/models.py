# 7. models/models.py - Complete U-Net architecture from the paper

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Compute sinusoidal embeddings for diffusion timesteps.
    
    Args:
        timesteps: Tensor of timestep indices
        dim: Embedding dimension (should be even)
    
    Returns:
        Embeddings of shape (batch_size, dim)
    """
    device = timesteps.device
    half_dim = dim // 2
    
    # Create frequency scaling
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half_dim, device=device).float() / half_dim)
    
    # Compute sinusoidal arguments
    args = timesteps[:, None].float() * freqs[None, :]
    
    # Compute embeddings
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    
    # Handle odd dimensions
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1), "constant", 0)
    
    return emb


class ResidualBlock(nn.Module):
    """
    Residual block with GroupNorm, convolution, and time-step embedding injection.
    
    This is the core building block described in Section V of the paper.
    """
    
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()
        
        # First conv path
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # Time embedding projection
        self.time_proj = nn.Linear(time_emb_dim, out_channels)
        
        # Second conv path
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Activation function
        self.activation = nn.SiLU()
        
        # Shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with time embedding injection.
        
        Args:
            x: Input tensor (B, in_channels, length)
            time_emb: Time embedding (B, time_emb_dim)
        
        Returns:
            Output tensor (B, out_channels, length)
        """
        # First path
        h = self.norm1(x)
        h = self.activation(h)
        h = self.conv1(h)
        
        # Inject time embedding
        emb = self.time_proj(time_emb)  # (B, out_channels)
        h = h + emb[:, :, None]  # Broadcast along length dimension
        
        # Second path
        h = self.norm2(h)
        h = self.activation(h)
        h = self.conv2(h)
        
        # Residual connection
        return self.shortcut(x) + h


class AttentionBlock(nn.Module):
    """
    Self-attention block for 1D signals.
    
    Enables the model to capture long-range dependencies as described in the paper.
    """
    
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels, 
            num_heads=num_heads,
            batch_first=True
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through self-attention.
        
        Args:
            x: Input tensor (B, C, L)
        
        Returns:
            Output tensor (B, C, L)
        """
        # Reshape for attention: (B, C, L) -> (B, L, C)
        B, C, L = x.size()
        h = x.permute(0, 2, 1)
        
        # Apply layer norm
        h = self.norm(h)
        
        # Self-attention
        attn_out, _ = self.attn(h, h, h)
        
        # Residual connection
        h = attn_out + h
        
        # Reshape back: (B, L, C) -> (B, C, L)
        return h.permute(0, 2, 1)


class ImprovedUNet(nn.Module):
    """
    Improved U-Net for time-series diffusion with Residual and Attention Blocks.
    
    This implements the exact architecture described in Section V of the research paper.
    Architecture parameters match Table I exactly.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        channel_mults: tuple = (1, 2, 4, 8),
        num_heads: int = 4,
        time_emb_dim: int = 256
    ):
        super().__init__()
        
        # Initial convolution
        self.conv_in = nn.Conv1d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Encoder (down-sampling path)
        self.down_blocks = nn.ModuleList()
        self.attn_blocks_down = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        
        prev_ch = base_channels
        num_levels = len(channel_mults)
        
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            
            # Residual block
            self.down_blocks.append(
                ResidualBlock(prev_ch, out_ch, time_emb_dim=time_emb_dim)
            )
            
            # Attention block (only for channel multipliers >= 4)
            if mult >= 4:
                self.attn_blocks_down.append(AttentionBlock(out_ch, num_heads=num_heads))
            else:
                self.attn_blocks_down.append(None)
            
            # Downsampling (except for last level)
            if i != num_levels - 1:
                self.downsamples.append(
                    nn.Conv1d(out_ch, out_ch, kernel_size=4, stride=2, padding=1)
                )
            
            prev_ch = out_ch
        
        # Middle/Bottleneck blocks
        self.mid_block1 = ResidualBlock(prev_ch, prev_ch, time_emb_dim=time_emb_dim)
        self.mid_attn = AttentionBlock(prev_ch, num_heads=num_heads)
        self.mid_block2 = ResidualBlock(prev_ch, prev_ch, time_emb_dim=time_emb_dim)
        
        # Decoder (up-sampling path)
        self.up_transposes = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.attn_blocks_up = nn.ModuleList()
        
        reversed_mults = list(reversed(channel_mults[:-1]))  # Exclude the last level
        
        for mult in reversed_mults:
            curr_ch = prev_ch
            skip_ch = base_channels * mult
            
            # Transpose convolution for upsampling
            self.up_transposes.append(
                nn.ConvTranspose1d(curr_ch, skip_ch, kernel_size=4, stride=2, padding=1)
            )
            
            # Residual block (input channels = skip_ch * 2 due to concatenation)
            self.up_blocks.append(
                ResidualBlock(skip_ch * 2, skip_ch, time_emb_dim=time_emb_dim)
            )
            
            # Attention block (only for channel multipliers >= 4)
            if mult >= 4:
                self.attn_blocks_up.append(AttentionBlock(skip_ch, num_heads=num_heads))
            else:
                self.attn_blocks_up.append(None)
            
            prev_ch = skip_ch
        
        # Final output convolution
        self.conv_out = nn.Conv1d(base_channels, in_channels, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the U-Net.
        
        Args:
            x: Input noised signal (B, 1, 3000)
            t: Timestep indices (B,)
        
        Returns:
            Predicted noise (B, 1, 3000)
        """
        # Embed timesteps
        t_emb = self.time_mlp(sinusoidal_embedding(t, self.time_mlp[0].in_features))
        
        # Initial convolution
        h = self.conv_in(x)
        
        # Encoder with skip connections
        skip_connections = []
        
        for i, (down_block, attn_block, downsample) in enumerate(
            zip(self.down_blocks, self.attn_blocks_down, self.downsamples + [None])
        ):
            # Residual block
            h = down_block(h, t_emb)
            
            # Attention block (if present)
            if attn_block is not None:
                h = attn_block(h)
            
            # Store skip connection before downsampling
            if downsample is not None:
                skip_connections.append(h)
                h = downsample(h)
        
        # Middle blocks
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)
        
        # Decoder with skip connections
        for up_conv, up_block, attn_block in zip(
            self.up_transposes, self.up_blocks, self.attn_blocks_up
        ):
            # Get corresponding skip connection
            skip_feat = skip_connections.pop()
            
            # Upsample
            h = up_conv(h)
            
            # Handle potential size mismatches
            if h.size(-1) != skip_feat.size(-1):
                diff = skip_feat.size(-1) - h.size(-1)
                if diff > 0:
                    h = F.pad(h, (0, diff))
                else:
                    h = h[:, :, :skip_feat.size(-1)]
            
            # Concatenate skip connection
            h = torch.cat([h, skip_feat], dim=1)
            
            # Residual block
            h = up_block(h, t_emb)
            
            # Attention block (if present)
            if attn_block is not None:
                h = attn_block(h)
        
        # Final output
        return self.conv_out(h)
