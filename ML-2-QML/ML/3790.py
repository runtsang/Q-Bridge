"""Hybrid classical-quantum-inspired binary classifier using classical convolutional layers,
a classical quanvolution filter, and a dense head with sigmoid activation.

The architecture mirrors the original hybrid model but replaces the quantum expectation
layer with a standard linear layer, while retaining the convolutional backbone and the
quanvolution-inspired feature extractor to capture spatial patterns.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuanvolutionFilter(nn.Module):
    """Classical 2×2 convolutional filter that emulates the quanvolution concept."""

    def __init__(self) -> None:
        super().__init__()
        # 1‑channel input → 4 output channels (2×2 patch → 4 features)
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.conv(x)
        return features.view(x.size(0), -1)  # flatten to (B, 4×14×14)


class HybridQuanvolutionNet(nn.Module):
    """
    Classical hybrid network that combines:
      • A standard convolutional branch (mirroring the original model)
      • A classical quanvolution filter branch
      • A dense head ending in a sigmoid for binary probability
    """

    def __init__(self) -> None:
        super().__init__()
        # Convolutional branch
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Quanvolution branch
        self.qfilter = QuanvolutionFilter()

        # Fully‑connected head
        # 540 (conv) + 784 (quanvolution) = 1324 features
        self.fc1 = nn.Linear(1324, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Conv branch
        y = F.relu(self.conv1(x))
        y = self.pool(y)
        y = self.drop1(y)
        y = F.relu(self.conv2(y))
        y = self.pool(y)
        y = self.drop1(y)
        y = torch.flatten(y, 1)

        # Quanvolution branch (expects 1‑channel input)
        q = self.qfilter(x.mean(dim=1, keepdim=True))

        # Concatenate features
        combined = torch.cat([y, q], dim=1)

        # Fully‑connected layers
        out = F.relu(self.fc1(combined))
        out = self.drop2(out)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)

        # Sigmoid output for binary classification
        probs = torch.sigmoid(out)
        return torch.cat([probs, 1 - probs], dim=-1)


__all__ = ["QuanvolutionFilter", "HybridQuanvolutionNet"]
