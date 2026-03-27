import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.AddModules.CAF import CAF
from ultralytics.nn.AddModules.DMRF import DMRF
from ultralytics.nn.AddModules.MDAFP import MDAFP


class MOESelector(nn.Module):
    """
    Mixture-of-Experts selector for fusion modules
    Experts:
        1. CAF
        2. DMRF
        3. MDAFP
    """

    def __init__(self, dim):

        super().__init__()

        # experts
        self.caf = CAF(dim)
        self.dmrf = DMRF(dim)
        self.mdafp = MDAFP(dim, 8, True)

        # gating network
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim * 2, dim // 4, 1),
            nn.SiLU(),
            nn.Conv2d(dim // 4, 3, 1)
        )

    def forward(self, x):

        rgb, ir = x

        fuse = torch.cat([rgb, ir], dim=1)

        # compute experts
        out1 = self.caf([rgb, ir])
        out2 = self.dmrf([rgb, ir])
        out3 = self.mdafp([rgb, ir])

        # gating weights
        gate_logits = self.gate(fuse).flatten(1)
        gate = F.softmax(gate_logits, dim=1)

        w1 = gate[:, 0].view(-1, 1, 1, 1)
        w2 = gate[:, 1].view(-1, 1, 1, 1)
        w3 = gate[:, 2].view(-1, 1, 1, 1)

        out = w1 * out1 + w2 * out2 + w3 * out3

        return out