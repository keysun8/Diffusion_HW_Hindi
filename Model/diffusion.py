import torch
import torch.nn as nn
import torch.nn.functional as F

from .style_encoder import StyleEncoder
from .content_encoder import ContentEncoder
from .blender import Blender
from .unet import TimeEmbedding, Unet

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
    def __init__(self, font_path="NotoSansDevanagari-Regular.ttf"):
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

        # Decouple style gradients from diffusion loss
        ver_map = ver_map.detach()
        hor_map = hor_map.detach()

        Q = self.content_encoder(text)
        cond = self.blender(Q, ver_map, hor_map)
        time_emb = self.time_embedding(time)
        output = self.unet(latent, cond, time_emb)
        output = self.final(output)
        global_emb = self.embedding_head(output)

        return output, global_emb, ver_emb, hor_emb, global_style_emb
