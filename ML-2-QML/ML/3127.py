"""Hybrid classical model inspired by Quanvolution and Quantum‑NAT."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassicalQuanvolutionFilter(nn.Module):
    """
    2×2 stride‑2 convolution that reduces a 28×28 image to 14×14 patches,
    each represented by 4 channels. The output is flattened for the head.
    """
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.conv(x)          # shape: (bsz, 4, 14, 14)
        return features.view(x.size(0), -1)  # shape: (bsz, 4*14*14)


class ClassicalHybridClassifier(nn.Module):
    """
    Classical head that maps the flattened patch features to 10 classes.
    Inspired by Quantum‑NAT’s fully‑connected projection and batch‑norm.
    """
    def __init__(self) -> None:
        super().__init__()
        self.filter = ClassicalQuanvolutionFilter()
        self.fc = nn.Sequential(
            nn.Linear(4 * 14 * 14, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
        self.norm = nn.BatchNorm1d(10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.filter(x)          # (bsz, 784)
        logits = self.fc(features)         # (bsz, 10)
        logits = self.norm(logits)         # (bsz, 10)
        return F.log_softmax(logits, dim=-1)


__all__ = ["ClassicalQuanvolutionFilter", "ClassicalHybridClassifier"]
