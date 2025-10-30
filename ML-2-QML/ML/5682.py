"""Hybrid classical estimator combining convolutional feature extraction and a fully‑connected regression head.

The architecture merges the compact feed‑forward design of EstimatorQNN with the convolutional
feature extractor from Quantum‑NAT, producing a 4‑dimensional embedding that can be directly
compared to the quantum baseline.

The network is fully PyTorch‑compatible and can be used interchangeably with the quantum
implementation for cross‑validation experiments.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridEstimatorQNN(nn.Module):
    """Convolutional backbone + lightweight fully‑connected head.

    The model first processes 2‑D inputs through a shallow CNN, then projects the 4‑dimensional
    feature map onto a regression output.  The 4‑D output mirrors the dimensionality of the
    quantum circuit’s measurement register, enabling direct comparison.
    """

    def __init__(self) -> None:
        super().__init__()
        # Feature extractor: two conv layers + pooling
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Flatten and project to 4 features
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),
        )
        # Optional scaling
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Input shape: (B, 1, 28, 28) typical MNIST
        features = self.features(x)
        flattened = features.view(features.size(0), -1)
        out = self.fc(flattened)
        return self.norm(out)


def EstimatorQNN() -> HybridEstimatorQNN:
    """Factory returning a classical HybridEstimatorQNN instance."""
    return HybridEstimatorQNN()


__all__ = ["HybridEstimatorQNN", "EstimatorQNN"]
