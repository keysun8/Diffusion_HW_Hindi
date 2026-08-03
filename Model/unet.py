import os
import math
import random

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch import nn
from torch.nn import functional as F
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms  
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from diffusers import AutoencoderKL
from torch.amp import GradScaler, autocast
from torchvision.utils import save_image
from torchmetrics.image.fid import FrechetInceptionDistance

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Time embedding + UNet 

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        args = t[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class SelfAttention(nn.Module):
    def __init__(self, channels, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = channels // n_heads
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x):
        b, c, h, w = x.shape
        x_flat = x.view(b, c, h * w).transpose(1, 2)
        qkv = self.qkv(x_flat)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(b, -1, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(b, -1, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(b, -1, self.n_heads, self.d_head).transpose(1, 2)
        attn = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.d_head), dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(b, h * w, c)
        out = self.proj(out)
        return out.transpose(1, 2).reshape(b, c, h, w)


class UNET_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim=512):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.residual_layer = (nn.Identity() if in_channels == out_channels
                                else nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0))

    def forward(self, feature, time):
        residue = feature
        feature = self.groupnorm_feature(feature)
        feature = F.silu(feature)
        feature = self.conv_feature(feature)
        time = F.silu(time)
        time = self.time_proj(time)
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        merged = self.groupnorm_merged(merged)
        merged = F.silu(merged)
        merged = self.conv_merged(merged)
        return merged + self.residual_layer(residue)

class CrossAttention_unet(nn.Module):
    def __init__(self, ch_1, ch_2=64):
        super().__init__()
        self.ch_1 = ch_1
        self.q = nn.Linear(ch_1, ch_1)
        self.k = nn.Linear(ch_2, ch_1)
        self.v = nn.Linear(ch_2, ch_1)
        self.proj = nn.Linear(ch_1, ch_1)

    def forward(self, x, cond):
        is_4d = x.dim() == 4
        
        # 1. If x is a 4D image, flatten it to a 3D sequence (B, H*W, C)
        if is_4d:
            b, c, h, w = x.shape
            x = x.view(b, c, h * w).transpose(1, 2)
            
        # 2. If cond is a 4D image (like the output of Blender), flatten it too
        if cond.dim() == 4:
            cond = cond.flatten(2).transpose(1, 2)

        # 3. Compute Q, K, V (both x and cond are now guaranteed to be 3D)
        q = self.q(x)
        k = self.k(cond)
        v = self.v(cond)

        # 4. Attention mechanism
        attn = torch.softmax(q @ k.transpose(1, 2) / math.sqrt(self.ch_1), dim=-1)
        out = attn @ v
        out = self.proj(out)

        # 5. If x started as 4D, unflatten the output back to 4D before returning
        if is_4d:
            out = out.transpose(1, 2).view(b, c, h, w)
            
        return out

class UNET_AttentionBlock(nn.Module):
    def __init__(self, channels, cond_dim=64, geglu_mult=2):
        super().__init__()
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(channels)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention_unet(channels, cond_dim)
        self.layernorm_3 = nn.LayerNorm(channels)
        hidden = geglu_mult * channels
        self.linear_geglu_1 = nn.Linear(channels, hidden * 2)
        self.linear_geglu_2 = nn.Linear(hidden, channels)
        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x, cond):
        residue_long = x
        x = self.conv_input(x)
        b, c, h, w = x.shape
        x = x.view((b, c, h * w)).transpose(1, 2)
        residue_short = x
        x = self.layernorm_1(x)
        x_spatial = x.transpose(1, 2).reshape(b, c, h, w)
        x_spatial = self.attention_1(x_spatial)
        x = x_spatial.reshape(b, c, h * w).transpose(1, 2)
        x += residue_short
        residue_short = x
        x = self.layernorm_2(x)
        x = self.attention_2(x, cond)
        x += residue_short
        residue_short = x
        x = self.layernorm_3(x)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)
        x = self.linear_geglu_2(x)
        x += residue_short
        x = x.transpose(1, 2).view((b, c, h, w))
        return self.conv_output(x) + residue_long


class SwitchSequential(nn.Sequential):
    def forward(self, x, cond, time):
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, cond)
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x


class width_Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=(1.0, 2.0), mode='nearest'))


class width_Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=(1, 2))

    def forward(self, x):
        return self.conv(x)


class Height_Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=(2.0, 1.0), mode='nearest'))


class Height_Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=(2, 1))

    def forward(self, x):
        return self.conv(x)


class Unet(nn.Module):
    def __init__(self, c1=224, c2=448, c3=896, time_dim=896, cond_dim=64, geglu_mult=2):
        super().__init__()

        def RB(i, o):
            return UNET_ResidualBlock(i, o, time_dim=time_dim)

        def AB(c):
            return UNET_AttentionBlock(c, cond_dim, geglu_mult=geglu_mult)

        self.encoders = nn.ModuleList([
            SwitchSequential(nn.Conv2d(4, c1, kernel_size=3, padding=1)),
            SwitchSequential(RB(c1, c1), AB(c1)),
            SwitchSequential(width_Downsample(c1)),
            SwitchSequential(RB(c1, c2), AB(c2)),
            SwitchSequential(width_Downsample(c2)),
            SwitchSequential(RB(c2, c3), AB(c3)),
            SwitchSequential(width_Downsample(c3)),
            SwitchSequential(Height_Downsample(c3)),
            SwitchSequential(RB(c3, c3), AB(c3)),
        ])
        self.bottleneck = SwitchSequential(RB(c3, c3), AB(c3), RB(c3, c3))
        self.decoders = nn.ModuleList([
            SwitchSequential(RB(c3 * 2, c3), AB(c3)),
            SwitchSequential(RB(c3 * 2, c3), Height_Upsample(c3)),
            SwitchSequential(RB(c3 * 2, c3), AB(c3), width_Upsample(c3)),
            SwitchSequential(RB(c3 * 2, c2), AB(c2)),
            SwitchSequential(RB(c2 * 2, c2), width_Upsample(c2)),
            SwitchSequential(RB(c2 * 2, c1), AB(c1)),
            SwitchSequential(RB(c1 * 2, c1), width_Upsample(c1)),
            SwitchSequential(RB(c1 * 2, c1), AB(c1)),
            SwitchSequential(RB(c1 * 2, c1)),
        ])

    def forward(self, x, cond, time):
        skip_connections = []
        for layers in self.encoders:
            x = layers(x, cond, time)
            skip_connections.append(x)
        x = self.bottleneck(x, cond, time)
        for layers in self.decoders:
            x = torch.cat((x, skip_connections.pop()), dim=1)
            x = layers(x, cond, time)
        return x
