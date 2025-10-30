"""Classical quanvolution network with learnable convolution and multi‑task head."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionNet(nn.Module):
    """Classical network mimicking the original quanvolution filter.
    It uses a learnable 2x2 convolution followed by a linear head.
    The architecture supports multiple output heads for multi‑task learning.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 10, heads: int = 1):
        super().__init__()
        # learnable 2x2 convolution with stride 2
        self.conv = nn.Conv2d(in_channels, 4, kernel_size=2, stride=2, bias=False)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        # adaptive pooling to match 14x14 patches
        self.pool = nn.AdaptiveAvgPool2d((14, 14))
        # linear heads for multi‑task classification
        self.heads = nn.ModuleList([nn.Linear(4 * 14 * 14, num_classes) for _ in range(heads)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        logits = [F.log_softmax(head(x), dim=-1) for head in self.heads]
        return torch.stack(logits, dim=1) if len(self.heads) > 1 else logits[0]
