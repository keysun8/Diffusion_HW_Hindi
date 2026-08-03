print("Hello start !!")

import os
import math
import random
import argparse

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Diffusion schedule constants (fixed: T )

TIMESTEPS = 1000
T = TIMESTEPS  

betas = torch.linspace(1e-4, 0.02, T, device=device)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)
sqrt_ab = torch.sqrt(alpha_bars)
sqrt_one_minus_ab = torch.sqrt(1.0 - alpha_bars)
sqrt_recip_a = torch.sqrt(1.0 / alphas)

betas_tilde = betas.clone()
betas_tilde[1:] = betas[1:] * (1.0 - alpha_bars[:-1]) / (1.0 - alpha_bars[1:])


def forward_diffusion(x0, t):
    """q(x_t | x_0). (Renamed/aliased from your `q_sample` -- `forward_diffusion`
    was being called in train_2() but never defined.)"""
    noise = torch.randn_like(x0)
    xt = (sqrt_ab[t, None, None, None] * x0
          + sqrt_one_minus_ab[t, None, None, None] * noise)
    return xt, noise

# Building blocks 

class Resblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        groups_in = min(32, in_channels) if in_channels >= 8 else in_channels
        groups_out = min(32, out_channels) if out_channels >= 8 else out_channels

        self.groupnorm_1 = nn.GroupNorm(groups_in, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.groupnorm_2 = nn.GroupNorm(groups_out, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        residual = x
        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv1(x)
        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x + self.residual_layer(residual)


class StyleBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            Resblock(128, 128), Resblock(128, 128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2),
            Resblock(128, 256), Resblock(256, 256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2),
            Resblock(256, 512), Resblock(512, 512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=2),
            Resblock(512, 512), Resblock(512, 512),
        )

    def forward(self, x):
        return self.net(x)


# Column / row masking -- gated on self.training, not requires_grad

def column_mask(feat_map, p=0.5):
    b, c, h, w = feat_map.shape
    mask = (torch.rand(b, 1, 1, w, device=feat_map.device) > p).float()
    return feat_map * mask


def row_mask(feat_map, p=0.5):
    b, c, h, w = feat_map.shape
    mask = (torch.rand(b, 1, h, 1, device=feat_map.device) > p).float()
    return feat_map * mask

class SelfAttentionBlock(nn.Module):
    """A purely self-attention block for feature extractors (no cross-attention condition needed)."""
    def __init__(self, channels):
        super().__init__()
        # Use 32 groups if channels >= 32, otherwise use channels
        groups = min(32, channels) if channels >= 8 else channels
        self.norm = nn.GroupNorm(groups, channels)
        self.attn = SelfAttention(channels)

    def forward(self, x):
        return x + self.attn(self.norm(x))

# Ver_Style / Hor_Style: these feed the Blender

class Ver_Style(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.res1 = Resblock(256, 256)
        self.res2 = Resblock(256, 64)
        # CHANGED: Use pure self-attention
        self.attn1 = SelfAttentionBlock(64)
        self.attn2 = SelfAttentionBlock(64)
        self.attn3 = SelfAttentionBlock(64)
        self.res3 = Resblock(64, 64)

    def forward(self, x):
        x = self.conv1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.attn1(x)
        x = self.attn2(x)
        x = self.attn3(x)
        x = self.res3(x)
        return x

class Hor_Style(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.res1 = Resblock(256, 256)
        self.res2 = Resblock(256, 64)
        self.attn1 = SelfAttentionBlock(64)
        self.attn2 = SelfAttentionBlock(64)
        self.attn3 = SelfAttentionBlock(64)
        self.res3 = Resblock(64, 64)

    def forward(self, x):
        x = self.conv1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.attn1(x)
        x = self.attn2(x)
        x = self.attn3(x)
        x = self.res3(x)
        return x


class StyleProxyHead(nn.Module):
    
    def __init__(self, in_channels=64, embed_dim=512, mode="column", mask_ratio=0.5):
        super().__init__()
        assert mode in ("column", "row")
        self.mode = mode
        self.mask_ratio = mask_ratio
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, embed_dim)

    def forward(self, feat_map):
        x = feat_map
        if self.training:
            b, c, h, w = x.shape
            if self.mode == "column":
                keep = (torch.rand(b, 1, 1, w, device=x.device) > self.mask_ratio).float()
            else:
                keep = (torch.rand(b, 1, h, 1, device=x.device) > self.mask_ratio).float()
            x = x * keep
            denom = keep.sum(dim=(2, 3), keepdim=True).clamp(min=1.0)
            x = x.sum(dim=(2, 3), keepdim=True) / denom   # true mean over kept entries
        else:
            x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)


from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class MobileNetStride8Backbone(nn.Module):
    def __init__(self, out_channels=512):
        super().__init__()
        mnet = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        self.features = mnet.features[:7]        # cumulative stride 8
        self.project = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.features(x)
        x = self.project(x)
        return x


class StyleEncoder(nn.Module):
  
    def __init__(self, embed_dim=512, mask_ratio=0.5):
        super().__init__()
        self.backbone = MobileNetStride8Backbone(out_channels=512)
        self.ver = Ver_Style()
        self.hor = Hor_Style()
        self.ver_proxy_head = StyleProxyHead(64, embed_dim, mode="column", mask_ratio=mask_ratio)
        self.hor_proxy_head = StyleProxyHead(64, embed_dim, mode="row", mask_ratio=mask_ratio)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, embed_dim)

    def forward(self, x):
        feat = self.backbone(x)

        ver_map = self.ver(feat)
        hor_map = self.hor(feat)

        ver_emb = self.ver_proxy_head(ver_map)
        hor_emb = self.hor_proxy_head(hor_map)

        global_emb = self.pool(feat).flatten(1)
        global_emb = self.fc(global_emb)
        global_emb = F.normalize(global_emb, p=2, dim=1)

        return ver_map, hor_map, ver_emb, hor_emb, global_emb

# Content encoder 

class UnifontTextEncoder(nn.Module):
    def __init__(self, font_path="/home/kishan/diffusion/NotoSansDevanagari-Regular.ttf", image_size=(64, 1024)):
        super().__init__()
        self.font_path = font_path
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])
        try:
            self.font = ImageFont.truetype(font_path, 28)
        except Exception:
            self.font = ImageFont.load_default()

    def render_text(self, text):
        H, W = self.image_size
        img = Image.new('RGB', (W, H), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        font_size = 28
        margin = 20
        min_font_size = 8

        # shrink font until text fits within canvas width
        font = self._load_font(font_size)
        while font_size > min_font_size:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            if text_w <= (W - margin):
                break
            font_size -= 2
            font = self._load_font(font_size)

        # vertical centering using actual glyph bbox (handles matras above
        # and subscript conjuncts/vowel signs below baseline correctly)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_h = bbox[3] - bbox[1]
        text_w = bbox[2] - bbox[0]

        x = max(0, (W - text_w) // 2 - bbox[0])
        y = max(0, (H - text_h) // 2 - bbox[1])

        draw.text((x, y), text, font=font, fill=(0, 0, 0))
        return img

    def _load_font(self, font_size):
        try:
            return ImageFont.truetype(
                self.font_path,
                font_size,
                layout_engine=ImageFont.Layout.RAQM,
            )
        except Exception:
            try:
                return ImageFont.truetype(self.font_path, font_size)
            except Exception:
                return self.font
    
    def forward(self, texts, device=None):
        imgs = [self.transform(self.render_text(t)) for t in texts]
        imgs = torch.stack(imgs)
        if device is not None:
            imgs = imgs.to(device)
        return imgs

class ContentEncoder(nn.Module):
    def __init__(self, font_path="/home/kishan/diffusion/NotoSansDevanagari-Regular.ttf"):
        super().__init__()
        self.unifont = UnifontTextEncoder(font_path)
        mnet = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        self.backbone = mnet.features[:7]
        self.proj = nn.Conv2d(32, 64, kernel_size=1)
        # CHANGED: Use pure self-attention so nn.Sequential doesn't crash
        self.attn_head = nn.Sequential(
            SelfAttentionBlock(64), 
            SelfAttentionBlock(64), 
            SelfAttentionBlock(64),
            Resblock(64, 64),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, 64, 1, 128))  # match feat map W=128
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, text):
        dev = next(self.parameters()).device
        x = self.unifont(text, device=dev)
        x = self.backbone(x)
        x = self.proj(x)
        x = x + self.pos_embed
        Q = self.attn_head(x)
        return Q

# Blender -- FIXED ORDER: horizontal first, then vertical (matches DiffBrush)

class Blender(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_ver = CrossAttention_unet(64, 64)
        self.self_ver_1 = SelfAttentionBlock(64)   
        self.self_ver_2 = SelfAttentionBlock(64)   
        self.cross_hor = CrossAttention_unet(64, 64)
        self.self_hor_1 = SelfAttentionBlock(64)   
        self.self_hor_2 = SelfAttentionBlock(64)  
        self.res_out = Resblock(64, 64)

    def forward(self, Q, S_ver, S_hor):
        cond = self.cross_ver(Q, S_ver)
        cond = self.self_ver_1(cond)
        cond = self.self_ver_2(cond)
        cond = self.cross_hor(cond, S_hor)
        cond = self.self_hor_1(cond)
        cond = self.self_hor_2(cond)
        cond = cond + Q * 0.5
        return self.res_out(cond)


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


# Proxy-NCA loss (your class -- correct as-is)

class ProxyNCALoss(nn.Module):
    def __init__(self, num_classes, embedding_dim, scale=32.0):
        super().__init__()
        self.proxies = nn.Parameter(torch.randn(num_classes, embedding_dim) * 0.01)
        self.scale = scale

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, p=2, dim=1)
        proxies = F.normalize(self.proxies, p=2, dim=1)
        dist = torch.cdist(embeddings, proxies) ** 2
        logits = -self.scale * dist
        return F.cross_entropy(logits, labels)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
class GlobalEmbeddingHead(nn.Module):
    def __init__(self, in_channels=4, embed_dim=512):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, embed_dim)

    def forward(self, x):
        x = self.pool(x).flatten(1)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)


class UNET_OutputLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.groupnorm(x)
        x = F.silu(x)
        return self.conv(x)


class Diffusion(nn.Module):
    def __init__(self, font_path="/home/kishan/diffusion/NotoSansDevanagari-Regular.ttf"):
        super().__init__()
        self.style_encoder = StyleEncoder()
        self.content_encoder = ContentEncoder(font_path)
        self.blender = Blender()
        self.time_embedding = TimeEmbedding(896)
        self.unet = Unet()
        self.final = UNET_OutputLayer(224, 4)
        self.embedding_head = GlobalEmbeddingHead(in_channels=4, embed_dim=512)

    def forward(self, latent, style_img, text, time):
        ver_map, hor_map, ver_emb, hor_emb, global_style_emb = self.style_encoder(style_img)

        # stop the main diffusion MSE gradient from flowing into the style
        # encoder through the conditioning path -- style encoder is trained
        # only via its own proxy losses (see train loop)
        ver_map = ver_map.detach()
        hor_map = hor_map.detach()

        Q = self.content_encoder(text)
        cond = self.blender(Q, ver_map, hor_map)
        time_emb = self.time_embedding(time)
        output = self.unet(latent, cond, time_emb)
        output = self.final(output)
        global_emb = self.embedding_head(output)

        return output, global_emb, ver_emb, hor_emb, global_style_emb


# Dataset 

class HandwritingProxyNCALatentDataset(Dataset):
    def __init__(self, root_dir, latent_ext=".pt"):
        self.samples = []
        self.writer_to_id = {}
        self.latent_ext = latent_ext

        writers = sorted(os.listdir(root_dir))
        writer_idx = 0

        for writer in writers:
            writer_path = os.path.join(root_dir, writer)
            if not os.path.isdir(writer_path):
                continue
            label_path = os.path.join(writer_path, "labels.txt")
            if not os.path.exists(label_path):
                continue
            with open(label_path, "r", encoding="utf-8") as f:
                texts = [line.strip() for line in f if line.strip()]

            latent_files = sorted(
                [f for f in os.listdir(writer_path) if f.endswith(self.latent_ext)],
                key=lambda x: int(os.path.splitext(x)[0]),
            )
            if not latent_files or not texts:
                continue
            if len(latent_files) != len(texts):
                print(f"[WARNING] skipping {writer}: latents ({len(latent_files)}) "
                      f"and labels ({len(texts)}) mismatch")
                continue

            self.writer_to_id[writer] = writer_idx
            writer_idx += 1

            for latent_name, txt in zip(latent_files, texts):
                self.samples.append({
                    "latent_path": os.path.join(writer_path, latent_name),
                    "text": txt,
                    "writer_id": self.writer_to_id[writer],
                })

        print(f"Total writers: {len(self.writer_to_id)}")
        print(f"Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self.latent_ext == ".pt":
            latent = torch.load(sample["latent_path"], weights_only=True)
        elif self.latent_ext == ".npy":
            latent = torch.from_numpy(np.load(sample["latent_path"]))
        else:
            raise ValueError(f"Unsupported latent extension: {self.latent_ext}")
        latent = latent.float()
        return latent, sample["text"], sample["writer_id"]


# ---------------------------------------------------------------------------
# Checkpointing (fixed key mismatch between save/load)
# ---------------------------------------------------------------------------
def save_checkpoint(epoch, model_diffusion, proxy_losses, optimizer_style,
                     optimizer_diffusion, scaler_style, scaler_diffusion,
                     loss_history, path="diff_checkpoints_5"):
    os.makedirs(path, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "style_encoder": model_diffusion.style_encoder.state_dict(),
        "content_encoder": model_diffusion.content_encoder.state_dict(),
        "blender": model_diffusion.blender.state_dict(),
        "unet": model_diffusion.unet.state_dict(),
        "final": model_diffusion.final.state_dict(),
        "embedding_head": model_diffusion.embedding_head.state_dict(),
        "ver_proxy_loss": proxy_losses["ver"].state_dict(),
        "hor_proxy_loss": proxy_losses["hor"].state_dict(),
        "global_proxy_loss": proxy_losses["global"].state_dict(),
        "optimizer_style": optimizer_style.state_dict(),
        "optimizer_diffusion": optimizer_diffusion.state_dict(),
        "scaler_style": scaler_style.state_dict(),
        "scaler_diffusion": scaler_diffusion.state_dict(),
        "loss_history": loss_history,
    }
    torch.save(ckpt, f"{path}/ckpt_epoch_{epoch}.pt")
    print(f"[CHECKPOINT SAVED] epoch {epoch}")


def load_checkpoint(path, model_diffusion, proxy_losses, optimizer_style,
                     optimizer_diffusion, scaler_style, scaler_diffusion):
    ckpt = torch.load(path, map_location=device)
    model_diffusion.style_encoder.load_state_dict(ckpt["style_encoder"])
    model_diffusion.content_encoder.load_state_dict(ckpt["content_encoder"])
    model_diffusion.blender.load_state_dict(ckpt["blender"])
    model_diffusion.unet.load_state_dict(ckpt["unet"])
    model_diffusion.final.load_state_dict(ckpt["final"])
    model_diffusion.embedding_head.load_state_dict(ckpt["embedding_head"])
    proxy_losses["ver"].load_state_dict(ckpt["ver_proxy_loss"])
    proxy_losses["hor"].load_state_dict(ckpt["hor_proxy_loss"])
    proxy_losses["global"].load_state_dict(ckpt["global_proxy_loss"])
    optimizer_style.load_state_dict(ckpt["optimizer_style"])
    optimizer_diffusion.load_state_dict(ckpt["optimizer_diffusion"])
    scaler_style.load_state_dict(ckpt["scaler_style"])
    scaler_diffusion.load_state_dict(ckpt["scaler_diffusion"])
    print(f"[CHECKPOINT LOADED] epoch {ckpt['epoch']}")
    return ckpt["epoch"], ckpt.get("loss_history", {})


# Loss-curve plotting

def plot_losses(loss_history, out_path="loss_curves.png"):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    keys_axes = [
        ("diffusion_mse", axes[0, 0], "Diffusion MSE"),
        ("style_ver_nca", axes[0, 1], "Vertical Proxy-NCA"),
        ("style_hor_nca", axes[1, 0], "Horizontal Proxy-NCA"),
        ("global_nca", axes[1, 1], "Global Proxy-NCA"),
    ]
    for key, ax, title in keys_axes:
        if loss_history.get(key):
            ax.plot(loss_history[key])
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# Image generation each epoch
@torch.no_grad()
def generate_images(model, vae, style_folder, texts, device, epoch, step=None, out_dir="eval_generated_v5"):
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((64, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")
    style_imgs_files = sorted(
        f for f in os.listdir(style_folder)
        if f.lower().endswith(IMG_EXTS)
    )[:len(texts)]

    if len(style_imgs_files) < len(texts):                      # <-- ADD THIS
        print(f"[WARN] only {len(style_imgs_files)} images in {style_folder}, "
              f"fewer than {len(texts)} requested test_texts")

    os.makedirs(out_dir, exist_ok=True)
    
    generated, reals = [], []

    for i, (img_name, text) in enumerate(zip(style_imgs_files, texts)):
        img_path = os.path.join(style_folder, img_name)
        style_img_pil = Image.open(img_path).convert("RGB")
        style_img = transform(style_img_pil).unsqueeze(0).to(device)
        reals.append(style_img.squeeze(0).cpu())

        # --- conditioning doesn't depend on t: compute it ONCE ---
        with autocast(device_type=device.type):
            ver_map, hor_map, _, _, _ = model.style_encoder(style_img)
            Q = model.content_encoder([text])
            cond = model.blender(Q, ver_map, hor_map)

        latent = torch.randn(1, 4, 8, 128, device=device)

        for t in reversed(range(0, TIMESTEPS)):
            t_tensor = torch.tensor([t], device=device)
            with autocast(device_type=device.type):
                time_emb = model.time_embedding(t_tensor)
                pred_noise = model.unet(latent, cond, time_emb)
                pred_noise = model.final(pred_noise)

            alpha, alpha_bar, beta = alphas[t], alpha_bars[t], betas[t]
            latent = (1 / torch.sqrt(alpha)) * (
                latent - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * pred_noise
            )
            if t > 0:
                latent = latent + torch.sqrt(beta) * torch.randn_like(latent)
            latent = torch.clamp(latent, -3.0, 3.0)

        latent = latent / 0.18215
        with autocast(device_type=device.type):
            img = vae.decode(latent).sample
        generated.append(img.squeeze(0).cpu())

        style_name = os.path.basename(os.path.normpath(style_folder))

        if step is not None:
            save_image((img.clamp(-1, 1) + 1) / 2,
                        f"{out_dir}/epoch_{epoch}_step_{step}_style_{style_name}_{i}.png")
        else:
            save_image((img.clamp(-1, 1) + 1) / 2,
                        f"{out_dir}/epoch_{epoch}_style_{style_name}_{i}.png")

    model.train()
    return reals, generated


def pick_random_style_folder(base_dir, folder_range):
    lo, hi = folder_range
    idx = random.randint(lo, hi)
    return os.path.join(base_dir, str(idx))


def train_2(model_diffusion, dataset, loader, vae, num_epochs=10,
            lambda_ver_nca=0.8, lambda_hor_nca=0.8, lambda_global_nca=0,
            checkpoint_every=5, resume_from=None, checkpoint_dir="diff_checkpoints_5",
            style_base_dir="/home/kishan/diffusion/output_dataset_hindi_with_json_line/test",
            style_folder_range=(1, 75),
            test_texts=(
                "नमस्ते धन्यवाद स्वागत", "मैं किशन हूँ।", "भाड़ में जाओ",
                "मैं अभी भी ड्राइवर हूँ।", "आज मौसम अच्छा है", "मुझे भूख लगी है",
            )):

    mse_fn = nn.MSELoss()
    proxy_losses = {
        "ver": ProxyNCALoss(num_classes=len(dataset.writer_to_id), embedding_dim=512).to(device),
        "hor": ProxyNCALoss(num_classes=len(dataset.writer_to_id), embedding_dim=512).to(device),
        "global": ProxyNCALoss(num_classes=len(dataset.writer_to_id), embedding_dim=512).to(device),
    }

    optimizer_style = torch.optim.AdamW(
        list(model_diffusion.style_encoder.parameters())
        + list(proxy_losses["ver"].parameters())
        + list(proxy_losses["hor"].parameters()),
        lr=1e-5, weight_decay=1e-4, betas=(0.9, 0.99),
    )

    # LR SPLIT: content_encoder + blender train faster (fresh-init, need to
    # learn glyph binding), unet/final/embedding_head keep the slower rate
    # they were already stable at.
    optimizer_diffusion = torch.optim.AdamW([
        {"params": model_diffusion.content_encoder.parameters(), "lr": 5e-5},
        {"params": model_diffusion.blender.parameters(), "lr": 5e-5},
        {"params": model_diffusion.unet.parameters(), "lr": 1e-5},
        {"params": model_diffusion.final.parameters(), "lr": 1e-5},
        {"params": model_diffusion.embedding_head.parameters(), "lr": 1e-5},
        {"params": proxy_losses["global"].parameters(), "lr": 1e-5},
    ], weight_decay=1e-4, betas=(0.9, 0.99))

    scaler_style = GradScaler()
    scaler_diffusion = GradScaler()

    loss_history = {"diffusion_mse": [], "style_ver_nca": [], "style_hor_nca": [], "global_nca": []}
    start_epoch = 0

    if resume_from is not None and os.path.exists(resume_from):
        # Manual load: model + proxy weights + style optimizer/scalers carry
        # over fine (unchanged structure). optimizer_diffusion is NOT loaded
        # from checkpoint -- its param groups changed (LR split), so it
        # starts fresh. Adam momentum/variance for the diffusion side resets;
        # weights themselves are untouched.
        ckpt = torch.load(resume_from, map_location=device)
        model_diffusion.style_encoder.load_state_dict(ckpt["style_encoder"])
        model_diffusion.content_encoder.load_state_dict(ckpt["content_encoder"])
        model_diffusion.blender.load_state_dict(ckpt["blender"])
        model_diffusion.unet.load_state_dict(ckpt["unet"])
        model_diffusion.final.load_state_dict(ckpt["final"])
        model_diffusion.embedding_head.load_state_dict(ckpt["embedding_head"])
        proxy_losses["ver"].load_state_dict(ckpt["ver_proxy_loss"])
        proxy_losses["hor"].load_state_dict(ckpt["hor_proxy_loss"])
        proxy_losses["global"].load_state_dict(ckpt["global_proxy_loss"])
        optimizer_style.load_state_dict(ckpt["optimizer_style"])
        scaler_style.load_state_dict(ckpt["scaler_style"])
        scaler_diffusion.load_state_dict(ckpt["scaler_diffusion"])

        start_epoch = ckpt["epoch"] + 1
        loss_history.update(ckpt.get("loss_history", {}))
        loss_history.pop("fid", None)  # drop stale FID history from old checkpoints, if present

        completed = len(loss_history.get("diffusion_mse", []))
        print(f"[RESUME] checkpoint epoch {ckpt['epoch']} loaded "
              f"({completed} epochs of history) | "
              f"resuming at epoch {start_epoch}/{num_epochs} "
              f"({num_epochs - start_epoch} epochs remaining) | "
              f"optimizer_diffusion REINITIALIZED with LR split "
              f"(content_encoder/blender=5e-5, unet/final/embedding_head=1e-5)")

        if completed != start_epoch:
            print(f"[RESUME][WARNING] loss_history length ({completed}) "
                  f"doesn't match resumed epoch index ({start_epoch}) — "
                  f"curves may be misaligned if you changed num_epochs "
                  f"or skipped a save between runs.")

        if start_epoch >= num_epochs:
            print(f"[RESUME] start_epoch ({start_epoch}) >= num_epochs "
                  f"({num_epochs}) — nothing to train, increase num_epochs.")

    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    for epoch in range(start_epoch, num_epochs):
        model_diffusion.train()
        pbar = tqdm(loader)

        running_mse, running_ver, running_hor, running_global, n_batches = 0.0, 0.0, 0.0, 0.0, 0

        for latents, texts, writer_ids in pbar:
            latents = latents.to(device)
            writer_ids = writer_ids.to(device)
            B = latents.shape[0]

            target_latents = latents
            with torch.no_grad():
                style_imgs = vae.decode(latents / 0.18215).sample

            t = torch.randint(0, TIMESTEPS, (B,), device=device).long()
            xt, noise = forward_diffusion(target_latents, t)

            # 1. STYLE ENCODER STEP -- vertical + horizontal proxy losses
            
            optimizer_style.zero_grad(set_to_none=True)

            with autocast(device_type=device.type):
                _, _, ver_emb, hor_emb, _ = model_diffusion.style_encoder(style_imgs)

                if torch.isnan(ver_emb).any() or torch.isnan(hor_emb).any():
                    print("NaN in ver_emb/hor_emb, skipping batch")
                    continue

                ver_emb = torch.clamp(ver_emb, -10, 10)
                hor_emb = torch.clamp(hor_emb, -10, 10)

                style_ver_loss = proxy_losses["ver"](ver_emb, writer_ids)
                style_hor_loss = proxy_losses["hor"](hor_emb, writer_ids)
                style_loss_total = lambda_ver_nca * style_ver_loss + lambda_hor_nca * style_hor_loss

            if torch.isnan(style_loss_total):
                print("NaN style_loss_total, skipping batch")
                continue

            scaler_style.scale(style_loss_total).backward()
            scaler_style.unscale_(optimizer_style)
            torch.nn.utils.clip_grad_norm_(model_diffusion.style_encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(proxy_losses["ver"].parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(proxy_losses["hor"].parameters(), 1.0)
            scaler_style.step(optimizer_style)
            scaler_style.update()

            # 2. DIFFUSION UNET STEP
            
            optimizer_diffusion.zero_grad(set_to_none=True)

            with torch.no_grad():
                ver_map, hor_map, _, _, _ = model_diffusion.style_encoder(style_imgs)

            with autocast(device_type=device.type):
                Q = model_diffusion.content_encoder(texts)
                cond = model_diffusion.blender(Q, ver_map, hor_map)
                time_emb = model_diffusion.time_embedding(t)
                pred_noise = model_diffusion.unet(xt, cond, time_emb)
                pred_noise = model_diffusion.final(pred_noise)
                global_emb = model_diffusion.embedding_head(pred_noise.detach())

                if torch.isnan(pred_noise).any() or torch.isnan(global_emb).any():
                    print("NaN in pred_noise/global_emb, skipping batch")
                    continue

                global_emb = torch.clamp(global_emb, -10, 10)
                diffusion_loss = mse_fn(pred_noise.float(), noise.float())
                global_nca_loss = proxy_losses["global"](global_emb, writer_ids)
                diffusion_total_loss = diffusion_loss + lambda_global_nca * global_nca_loss

            if torch.isnan(diffusion_total_loss):
                print("NaN diffusion_total_loss, skipping batch")
                continue

            scaler_diffusion.scale(diffusion_total_loss).backward()
            scaler_diffusion.unscale_(optimizer_diffusion)
            for module in (model_diffusion.content_encoder, model_diffusion.blender,
                           model_diffusion.unet, model_diffusion.final, model_diffusion.embedding_head):
                torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
            scaler_diffusion.step(optimizer_diffusion)
            scaler_diffusion.update()

            running_mse += diffusion_loss.item()
            running_ver += style_ver_loss.item()
            running_hor += style_hor_loss.item()
            running_global += global_nca_loss.item()
            n_batches += 1

            pbar.set_description(
                f"Epoch {epoch+1} | Diff: {diffusion_loss.item():.4f} | "
                f"Ver: {style_ver_loss.item():.4f} | Hor: {style_hor_loss.item():.4f} | "
                f"Global: {global_nca_loss.item():.4f}"
            )

            if n_batches % 500 == 0:
                preview_style_folder = pick_random_style_folder(style_base_dir, style_folder_range)
                print(f"\n[Mid-epoch Generation] Epoch {epoch}, Step {n_batches}... "
                      f"using style folder: {preview_style_folder}")
                generate_images(
                    model_diffusion, vae, style_folder=preview_style_folder,
                    texts=list(test_texts), device=device, epoch=epoch, step=n_batches
                )

        n_batches = max(n_batches, 1)
        loss_history["diffusion_mse"].append(running_mse / n_batches)
        loss_history["style_ver_nca"].append(running_ver / n_batches)
        loss_history["style_hor_nca"].append(running_hor / n_batches)
        loss_history["global_nca"].append(running_global / n_batches)

        preview_style_folder = pick_random_style_folder(style_base_dir, style_folder_range)
        print(f"[Epoch Generation] preview style folder: {preview_style_folder}")
        generate_images(
            model_diffusion, vae, style_folder=preview_style_folder,
            texts=list(test_texts), device=device, epoch=epoch,
        )

        plot_losses(loss_history)

        if (epoch + 1) % checkpoint_every == 0:
            save_checkpoint(
                epoch, model_diffusion, proxy_losses, optimizer_style,
                optimizer_diffusion, scaler_style, scaler_diffusion, loss_history,
                path=checkpoint_dir,
            )

    return loss_history

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Params     : {total_params:,}")
    print(f"Trainable Params : {trainable_params:,}")
    return total_params, trainable_params


def parse_args():
    parser = argparse.ArgumentParser(description="Train handwriting diffusion model")
    parser.add_argument("--data_root", type=str,
                         default="/home/kishan/diffusion/output_dataset_hindi_with_json_line_latent/train",
                         help="Root directory of the latent training dataset")
    parser.add_argument("--latent_ext", type=str, default=".pt",
                         help="File extension of stored latents (.pt or .npy)")
    parser.add_argument("--vae_path", type=str, default="/home/kishan/diffusion/vae",
                         help="Path to the pretrained AutoencoderKL VAE")
    parser.add_argument("--batch_size", type=int, default=8,
                         help="Training batch size")
    parser.add_argument("--num_workers", type=int, default=4,
                         help="DataLoader worker count")
    parser.add_argument("--num_epochs", type=int, default=800,
                         help="Total number of epochs to train for")
    parser.add_argument("--checkpoint_every", type=int, default=5,
                         help="Save a checkpoint every N epochs")
    parser.add_argument("--checkpoint_dir", type=str, default="diff_checkpoints_5",
                         help="Directory to save checkpoints to")
    parser.add_argument("--resume_ckpt", type=str, default=None,
                         help="Path to a checkpoint .pt file to resume training from")
    parser.add_argument("--style_base_dir", type=str,
                         default="/home/kishan/diffusion/output_dataset_hindi_with_json_line/test",
                         help="Base directory containing per-writer style-image folders")
    parser.add_argument("--font_path", type=str,
                         default="/home/kishan/diffusion/NotoSansDevanagari-Regular.ttf",
                         help="Path to the Devanagari .ttf font used for content rendering")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    dataset = HandwritingProxyNCALatentDataset(
        args.data_root,
        latent_ext=args.latent_ext,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                         num_workers=args.num_workers, pin_memory=True)

    vae = AutoencoderKL.from_pretrained(args.vae_path).to(device)
    model_diffusion = Diffusion(font_path=args.font_path).to(device)

    train_2(
        model_diffusion, dataset, loader, vae,
        num_epochs=args.num_epochs,
        checkpoint_every=args.checkpoint_every,
        resume_from=args.resume_ckpt,
        checkpoint_dir=args.checkpoint_dir,
        style_base_dir=args.style_base_dir,
    )
