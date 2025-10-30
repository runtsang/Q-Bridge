"""Hybrid classical CNN that fuses a 2×2 quantum‑style filter with a compact CNN backbone."""

from __future__ import annotations

import torch
from torch import nn


class ConvFilter2x2(nn.Module):
    """Custom 2×2 convolutional filter emulating the quantum filter behavior."""
    def __init__(self, threshold: float = 0.0) -> None:
        super().__init__()
        self.threshold = threshold
        # Use padding=1 so output spatial size matches input
        self.conv = nn.Conv2d(1, 1, kernel_size=2, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations


class ConvGen304(nn.Module):
    """Classical convolutional network that fuses a 2×2 quantum‑style filter with a
    small CNN backbone inspired by the Quantum‑NAT architecture."""
    def __init__(self, threshold: float = 0.0) -> None:
        super().__init__()
        self.filter = ConvFilter2x2(threshold=threshold)
        # The first conv now expects 2 input channels (original image + filter output)
        self.features = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply the quantum‑style filter on every 2×2 patch
        patch_out = self.filter(x)
        # Concatenate the filter output with the original image channel
        x = torch.cat([x, patch_out], dim=1)
        features = self.features(x)
        flattened = features.view(features.size(0), -1)
        out = self.fc(flattened)
        return self.norm(out)


__all__ = ["ConvGen304"]
