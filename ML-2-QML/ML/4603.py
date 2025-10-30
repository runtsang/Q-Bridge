"""Hybrid classical regression model combining EstimatorQNN and QuantumNAT ideas.

Features:
- Two‑layer CNN encoder (same as QFCModel)
- Fully‑connected regression head with a BatchNorm1d layer
- Simple interface: forward(inputs) -> torch.Tensor of shape (batch, 1)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EstimatorQNNRegression(nn.Module):
    """CNN → FC regression model inspired by EstimatorQNN and QuantumNAT."""

    def __init__(self) -> None:
        super().__init__()
        # Convolutional feature extractor (identical to QFCModel.features)
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Fully‑connected projection to a single output
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        # Batch‑norm on the scalar output (mirrors QFCModel.norm)
        self.norm = nn.BatchNorm1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (batch, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Normalised scalar predictions of shape (batch, 1).
        """
        bsz = x.shape[0]
        features = self.features(x)
        flat = features.view(bsz, -1)
        out = self.fc(flat)
        return self.norm(out)


__all__ = ["EstimatorQNNRegression"]
