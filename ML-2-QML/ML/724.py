"""Enhanced classical quanvolution filter with residual connections and dropout."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuanvolutionFilter(nn.Module):
    """Convolutional filter with optional residual shortcut and dropout."""
    def __init__(self, in_ch: int = 1, out_ch: int = 4, kernel_size: int = 2,
                 stride: int = 2, residual: bool = True, dropout: float = 0.0) -> None:
        super().__init__()
        self.residual = residual
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride)
        if residual:
            self.res_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out = self.conv(x)
        if self.residual:
            out = out + self.res_conv(x)
        out = self.dropout(out)
        return out.view(x.size(0), -1)


class QuanvolutionClassifier(nn.Module):
    """Hybrid classifier using the enhanced quanvolution filter."""
    def __init__(self, in_features: int = 4 * 14 * 14, num_classes: int = 10,
                 dropout: float = 0.3) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter(dropout=0.0)  # no dropout inside filter
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.qfilter(x)
        features = self.dropout(features)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]
