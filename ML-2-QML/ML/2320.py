"""Hybrid classical model combining quanvolution filter with a regression head.

The model can be used for classification (log‑softmax output) or regression
(continuous output) depending on the ``regression`` flag.  It reuses the
classical 2×2 quanvolution filter from the original seed and augments it
with a small MLP head inspired by the regression example.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuanvolutionFilter(nn.Module):
    """Simple 2×2 quanvolution filter producing 4×14×14 features."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)


class HybridQuanvolutionModel(nn.Module):
    """Hybrid quanvolution model with optional regression head."""
    def __init__(self, num_classes: int = 10, regression: bool = False) -> None:
        super().__init__()
        self.filter = QuanvolutionFilter()
        # Feature dimension: 4 * 14 * 14 = 784
        self.regression = regression
        if regression:
            # Regression head: 784 → 32 → 16 → 1
            self.head = nn.Sequential(
                nn.Linear(784, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            )
        else:
            # Classification head
            self.classifier = nn.Linear(784, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.filter(x)
        if self.regression:
            return self.head(features).squeeze(-1)
        else:
            logits = self.classifier(features)
            return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionFilter", "HybridQuanvolutionModel"]
