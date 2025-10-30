"""Combined CNN‑FC regressor inspired by EstimatorQNN and Quantum‑NAT.

This module fuses a lightweight convolutional front‑end from Quantum‑NAT
with a fully‑connected regression head, providing richer feature extraction
while remaining fully classical. The network accepts 2‑D inputs of shape
(batch, 1, 28, 28) and produces a single continuous output.
"""

from __future__ import annotations

import torch
from torch import nn

class EstimatorQNN(nn.Module):
    """Hybrid CNN‑FC regression network.

    The module first extracts 4 feature maps via two convolutional blocks,
    then projects them to a 64‑dimensional vector before the final
    regression head. Batch‑norm and ReLU activations are used throughout.
    """

    def __init__(self) -> None:
        super().__init__()
        # Feature extractor – borrowed from Quantum‑NAT
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Flattened feature size: 16 * 7 * 7 for 28x28 input
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        self.norm = nn.BatchNorm1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Forward pass.

        Parameters
        ----------
        x:
            Input tensor of shape (batch, 1, H, W).

        Returns
        -------
        torch.Tensor
            Predicted scalar per sample.
        """
        features = self.features(x)
        flattened = features.view(x.shape[0], -1)
        out = self.fc(flattened)
        return self.norm(out)

__all__ = ["EstimatorQNN"]
