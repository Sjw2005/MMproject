import torch
import torch.nn as nn
import torch.nn.functional as F


class CAF(nn.Module):
    """
    Cross Attention Fusion (CAF)

    RGB → Query
    IR  → Key / Value

    输出:
    IR-dominant fusion feature
    """

    def __init__(self, dim):
        super().__init__()

        self.dim = dim
        self.scale = dim ** -0.5

        self.q = nn.Conv2d(dim, dim, 1, bias=False)
        self.k = nn.Conv2d(dim, dim, 1, bias=False)
        self.v = nn.Conv2d(dim, dim, 1, bias=False)

        self.proj = nn.Conv2d(dim, dim, 1, bias=False)

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):

        rgb, ir = x

        B, C, H, W = rgb.shape

        q = self.q(rgb).reshape(B, C, -1)
        k = self.k(ir).reshape(B, C, -1)
        v = self.v(ir).reshape(B, C, -1)

        attn = torch.softmax((q.transpose(1, 2) @ k) * self.scale, dim=-1)

        out = (attn @ v.transpose(1, 2)).transpose(1, 2)
        out = out.reshape(B, C, H, W)

        out = self.proj(out) + ir

        # LayerNorm stabilization
        out = out.flatten(2).transpose(1, 2)
        out = self.norm(out)
        out = out.transpose(1, 2).reshape(B, C, H, W)

        return out