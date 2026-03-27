import torch
import torch.nn as nn
from einops import rearrange


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class BaseConv(nn.Module):
    """Ultralytics-like Conv"""
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class AvgMaxPool2d(nn.Module):
    """Better for IR salient targets"""
    def __init__(self, k=3, s=1, p=1):
        super().__init__()
        self.avg = nn.AvgPool2d(k, s, p)
        self.max = nn.MaxPool2d(k, s, p)

    def forward(self, x):
        return 0.5 * self.avg(x) + 0.5 * self.max(x)


class LAE_v2(nn.Module):
    """
    Improved LAE:
      - supports c1->c2 (Conv replacement)
      - supports stride=2 (your backbone Conv use-case)
      - more stable attention (channel-averaged att)
      - selectable pooling (RGB vs IR)
    """
    def __init__(self, c1, c2, k=3, s=2, group=8, pool_type="avg"):
        super().__init__()
        assert s == 2, "This LAE_v2 is designed for stride=2 downsample conv replacement."
        self.softmax = nn.Softmax(dim=-1)

        # attention pooling (rgb: avg, ir: avgmax)
        if pool_type == "avgmax":
            pool = AvgMaxPool2d(3, 1, 1)
        else:
            pool = nn.AvgPool2d(3, 1, 1)

        # attention: keep at c1 channels then reshape to 4
        self.attention = nn.Sequential(
            pool,
            BaseConv(c1, c1, k=1)
        )

        # sampler produces 4*c2 then reshape
        g = max(1, c1 // group)
        self.ds_conv = BaseConv(c1, c2 * 4, k=k, s=2, g=g)

        # optional projection for stability
        self.proj = BaseConv(c2, c2, k=1)

    def forward(self, x):
        # attention: (B,c1,2h,2w) -> (B,c1,h,w,4)
        att = rearrange(self.attention(x), "b c (s1 h) (s2 w) -> b c h w (s1 s2)", s1=2, s2=2)
        att = self.softmax(att)

        # stabilize: channel-average attention (reduce noisy per-channel softmax)
        att = att.mean(dim=1, keepdim=True)  # (B,1,h,w,4)

        # sampled: (B,4*c2,h,w) -> (B,c2,h,w,4)
        y = rearrange(self.ds_conv(x), "b (s c) h w -> b c h w s", s=4)

        y = torch.sum(y * att, dim=-1)  # (B,c2,h,w)
        return self.proj(y)


class GateNet(nn.Module):
    """Predict a gate scalar in [0,1] per sample based on global statistics."""
    def __init__(self, c1, hidden=None):
        super().__init__()
        if hidden is None:
            hidden = max(16, c1 // 16)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c1, hidden, 1, bias=True),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(self.pool(x))  # (B,1,1,1)


class MLA(nn.Module):
    """
    Modality-Adaptive LAE (Conv replacement):
    - Two experts: RGB-like and IR-like
    - Gate chooses mixture automatically
    - Same interface as Conv: (c1, c2, k, s)
    """
    def __init__(self, c1, c2, k=3, s=2, group=8):
        super().__init__()
        assert s == 2, "Backbone downsample conv in your YAML uses stride=2; this module targets that."

        # RGB expert: avg pooling attention
        self.lae_vis = LAE_v2(c1, c2, k=k, s=s, group=group, pool_type="avg")
        # IR expert: avg+max pooling attention (salient)
        self.lae_ir  = LAE_v2(c1, c2, k=k, s=s, group=group, pool_type="avgmax")

        self.gate = GateNet(c1)

    def forward(self, x):
        w = self.gate(x)  # (B,1,1,1)
        y_vis = self.lae_vis(x)
        y_ir  = self.lae_ir(x)
        return w * y_vis + (1.0 - w) * y_ir
