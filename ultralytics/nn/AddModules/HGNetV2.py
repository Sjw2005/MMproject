import torch.nn as nn

from ultralytics.nn.modules.block import HGBlock

__all__ = ("Light_HGBlock",)


class Light_HGBlock(HGBlock):
    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=True, shortcut=False, act=nn.ReLU()):
        super().__init__(c1, cm, c2, k=k, n=n, lightconv=bool(lightconv), shortcut=shortcut, act=act)
