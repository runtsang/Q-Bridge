"""Enhanced classical quanvolution with an attention mechanism."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvAttentionBlock(nn.Module):
    """Simple 2‑D attention applied to feature maps."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.attn = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        attn = torch.sigmoid(self.attn(x))
        return self.bn(x * attn)


class QuanvolutionPlus(nn.Module):
    """Purely classical implementation that mimics a quanvolution filter."""
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.patch_size = 2
        self.conv = nn.Conv2d(in_channels, 4, kernel_size=self.patch_size, stride=self.patch_size)
        self.attn_block = ConvAttentionBlock(4, 4)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 28, 28)
        patches = self.conv(x)                # (B, 4, 14, 14)
        patches = self.attn_block(patches)    # attention
        features = self.flatten(patches)      # (B, 4*14*14)
        logits = self.fc(features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionPlus"]
