"""Hybrid classical fraud detection model inspired by QCNN, Quanvolution, and Quantum‑NAT."""

from __future__ import annotations

import torch
from torch import nn


class FraudDetectionHybrid(nn.Module):
    """
    Classical feature extractor resembling a QCNN followed by a linear head.
    The architecture mirrors the depth of the quantum version but remains purely
    classical, making it a drop‑in replacement when a quantum backend is unavailable.
    """

    def __init__(self) -> None:
        super().__init__()
        # Feature mapping: 2‑dimensional input → 8‑dim latent space
        self.feature_map = nn.Sequential(nn.Linear(2, 8), nn.Tanh())
        # Convolution‑like layers (fully connected)
        self.conv1 = nn.Sequential(nn.Linear(8, 8), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(8, 6), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(6, 4), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(4, 3), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(3, 2), nn.Tanh())
        # Final classification head
        self.head = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the stacked feature extractor and head.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 2).

        Returns
        -------
        torch.Tensor
            Output probability of shape (batch, 1).
        """
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))


__all__ = ["FraudDetectionHybrid"]
