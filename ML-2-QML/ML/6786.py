"""
HybridEstimatorQNN – Classical implementation using PyTorch.
Implements a CNN backbone followed by a fully‑connected head.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class HybridEstimatorQNN(nn.Module):
    """
    Classical hybrid network:
        * Convolutional feature extractor (similar to QFCModel).
        * Fully‑connected projection to 4 output features.
        * Batch‑normalisation for stable training.
    """

    def __init__(self, in_channels: int = 1, out_features: int = 4) -> None:
        super().__init__()
        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Flatten and fully‑connected head
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, out_features),
        )
        self.norm = nn.BatchNorm1d(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass: extract features, flatten, project, and normalise.
        """
        bsz = x.shape[0]
        features = self.features(x)
        flattened = features.view(bsz, -1)
        out = self.fc(flattened)
        return self.norm(out)


def EstimatorQNN() -> HybridEstimatorQNN:
    """
    Factory function mirroring the original EstimatorQNN API.
    Returns an instance of the hybrid classical network.
    """
    return HybridEstimatorQNN()


__all__ = ["HybridEstimatorQNN", "EstimatorQNN"]
