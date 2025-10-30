"""Enhanced classical quanvolution with attention for richer feature extraction."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionFilter(nn.Module):
    """Classical 2×2 patch filter with depthwise separable conv and self‑attention."""

    def __init__(self, in_channels: int = 1, out_channels: int = 4) -> None:
        super().__init__()
        # Depthwise conv to preserve per‑channel information
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2, groups=in_channels, bias=False)
        # Pointwise conv to mix channels
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        # Self‑attention over patches
        self.attn = nn.MultiheadAttention(embed_dim=out_channels, num_heads=2, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_channels, H, W)
        patches = self.depthwise(x)  # (batch, in_channels, H/2, W/2)
        patches = self.pointwise(patches)  # (batch, out_channels, H/2, W/2)
        # Flatten spatial dims into sequence
        batch, channels, h, w = patches.shape
        seq = patches.view(batch, channels, h * w).transpose(1, 2)  # (batch, seq_len, channels)
        # Self‑attention
        attn_out, _ = self.attn(seq, seq, seq)
        # Flatten back to feature vector
        out = attn_out.transpose(1, 2).contiguous().view(batch, -1)
        return out

class QuanvolutionClassifier(nn.Module):
    """Classifier that uses the attention‑based quanvolution filter followed by a linear head."""

    def __init__(self, in_channels: int = 1, num_classes: int = 10) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter(in_channels, out_channels=4)
        # 4 (out_channels) * 14 * 14 (patches) flattened
        self.linear = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]
