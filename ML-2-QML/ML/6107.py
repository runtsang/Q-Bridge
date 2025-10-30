"""Purely classical hybrid model inspired by Quanvolution and Quantum‑NAT."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuanvolutionHybridModel(nn.Module):
    """Classical CNN backbone followed by a linear classifier."""
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(32, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.features(x)
        flattened = features.view(features.size(0), -1)
        logits = self.classifier(flattened)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionHybridModel"]
