"""Enhanced classical quanvolution filter with learnable weights and normalization."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionFilter(nn.Module):
    """Two‑by‑two learnable classical convolution with optional batch norm and ReLU."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(4)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)
        return out.view(x.size(0), -1)

class QuanvolutionClassifier(nn.Module):
    """Classifier head that follows the learnable quanvolution filter."""

    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]
