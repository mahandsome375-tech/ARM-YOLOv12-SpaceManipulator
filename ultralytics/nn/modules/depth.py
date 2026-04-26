import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv

class DepthHead(nn.Module):
    """A lightweight depth prediction head that takes P3 features as input and outputs single-channel depth.
       The output range is [0, 1] through Sigmoid; L1 or SmoothL1 can be used during training.
    """
    def __init__(self, in_ch=256, mid=128, out_ch=1):
        super().__init__()
        self.block = nn.Sequential(
            Conv(in_ch, mid, k=3, s=1),
            Conv(mid, mid, k=3, s=1),
            nn.Conv2d(mid, out_ch, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x, up_factor=8):

        d = self.block(x)

        d = F.interpolate(d, scale_factor=up_factor, mode="bilinear", align_corners=False)
        return d.clamp(0, 1).sigmoid()
