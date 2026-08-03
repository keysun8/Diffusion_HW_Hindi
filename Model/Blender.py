import torch
import torch.nn as nn
from .style_encoder import Resblock, SelfAttentionBlock
from .unet import CrossAttention_unet

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
