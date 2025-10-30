"""Enhanced classical Quanvolution filter with attention."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuanvolutionFilter(nn.Module):
    """
    Classical convolutional filter with learnable scaling and
    batch normalization. The filter outputs a 4‑channel feature map
    of size 14×14 for each input image.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2, bias=False)
        self.bn = nn.BatchNorm2d(4)
        self.scale = nn.Parameter(torch.ones(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        features = self.bn(features)
        features = features * self.scale.view(1, -1, 1, 1)
        return features  # shape: (batch, 4, 14, 14)


class QuanvolutionClassifier(nn.Module):
    """
    Hybrid classifier that applies a multi‑head self‑attention
    mechanism on the convolutional features before the final
    linear layer.
    """

    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.attn = nn.MultiheadAttention(embed_dim=4, num_heads=2, batch_first=True)
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convolutional features: (batch, 4, 14, 14)
        features = self.qfilter(x)
        # Reshape to (batch, seq_len, embed_dim)
        seq = features.permute(0, 2, 3, 1).reshape(x.size(0), -1, 4)
        attn_output, _ = self.attn(seq, seq, seq)
        attn_output = attn_output.reshape(x.size(0), -1)
        logits = self.linear(attn_output)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]
