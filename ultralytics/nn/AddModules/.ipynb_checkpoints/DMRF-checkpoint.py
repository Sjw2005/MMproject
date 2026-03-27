# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class DMRF(nn.Module):
#     """
#     Dual-Modality Reliability Fusion
#     IR-dominant cross attention fusion
#     """

#     def __init__(self, dim):
#         super().__init__()

#         self.q = nn.Conv2d(dim, dim, 1)
#         self.k = nn.Conv2d(dim, dim, 1)
#         self.v = nn.Conv2d(dim, dim, 1)

#         self.scale = dim ** -0.5

#         self.gate = nn.Sequential(
#             nn.Conv2d(dim * 2, dim, 1),
#             nn.SiLU(),
#             nn.Conv2d(dim, dim, 1),
#             nn.Sigmoid()
#         )

#         self.proj = nn.Conv2d(dim, dim, 1)

#     def forward(self, x):

#         rgb, ir = x

#         B,C,H,W = rgb.shape

#         q = self.q(rgb).reshape(B,C,-1)
#         k = self.k(ir).reshape(B,C,-1)
#         v = self.v(ir).reshape(B,C,-1)

#         attn = torch.softmax((q.transpose(1,2) @ k) * self.scale, dim=-1)

#         out = (attn @ v.transpose(1,2)).transpose(1,2).reshape(B,C,H,W)

#         gate = self.gate(torch.cat([rgb,ir],1))

#         fused = gate * ir + (1-gate) * out

#         return self.proj(fused) + ir

import torch
import torch.nn as nn
import torch.nn.functional as F


class DMRF(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.q = nn.Conv2d(dim, dim, 1)
        self.k = nn.Conv2d(dim, dim, 1)
        self.v = nn.Conv2d(dim, dim, 1)

        self.scale = dim ** -0.5

        self.gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.SiLU(),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )

        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):

        rgb, ir = x

        B,C,H,W = rgb.shape

        q = self.q(rgb).reshape(B,C,-1)
        k = self.k(ir).reshape(B,C,-1)
        v = self.v(ir).reshape(B,C,-1)

        attn = torch.softmax((q.transpose(1,2) @ k) * self.scale, dim=-1)

        out = (attn @ v.transpose(1,2)).transpose(1,2).reshape(B,C,H,W)

        gate = self.gate(torch.cat([rgb,ir],1))

        fused = gate * ir + (1-gate) * out

        return self.proj(fused) + ir