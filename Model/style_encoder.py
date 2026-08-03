import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

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


class SelfAttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        groups = min(32, channels) if channels >= 8 else channels
        self.norm = nn.GroupNorm(groups, channels)
        self.attn = SelfAttention(channels)

    def forward(self, x):
        return x + self.attn(self.norm(x))


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
        attn = torch.softmax(q @ k.transpose(-1, -2) / (self.d_head ** 0.5), dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(b, h * w, c)
        out = self.proj(out)
        return out.transpose(1, 2).reshape(b, c, h, w)


class Ver_Style(nn.Module):
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
            x = x.sum(dim=(2, 3), keepdim=True) / denom
        else:
            x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)


class MobileNetStride8Backbone(nn.Module):
    def __init__(self, out_channels=512):
        super().__init__()
        mnet = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        self.features = mnet.features[:7]
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
