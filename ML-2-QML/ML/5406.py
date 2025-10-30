"""Classical component of the hybrid EstimatorQNNGen297 model.

This module implements a lightweight feature extractor that mimics the
behaviour of the original EstimatorQNN while adding a convolutional
filter and a small feed‑forward network.  The extracted 1‑dimensional
feature is intended to be fed into the quantum head defined in the
corresponding QML module.
"""

from __future__ import annotations

import torch
from torch import nn
import numpy as np

# --------------------------------------------------------------------
# Classical Conv filter (from reference 2)
# --------------------------------------------------------------------
class ConvFilter(nn.Module):
    """Simple 2‑D filter that emulates the quantum quanvolution layer."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, H, W)
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        # Global average pooling to a scalar per sample
        return activations.mean(dim=[2, 3])  # (batch, 1)

# --------------------------------------------------------------------
# Feed‑forward regressor (from reference 1)
# --------------------------------------------------------------------
class EstimatorNN(nn.Module):
    """Small fully‑connected network that maps a 2‑dimensional input
    to a single output value."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# --------------------------------------------------------------------
# Feature extractor that chains the filter and the regressor
# --------------------------------------------------------------------
class EstimatorQNNGen297(nn.Module):
    """Combines a convolutional filter and a feed‑forward regressor
    to produce a 1‑dimensional feature vector."""
    def __init__(self, kernel_size: int = 2) -> None:
        super().__init__()
        self.conv = ConvFilter(kernel_size=kernel_size)
        self.regressor = EstimatorNN()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, H, W)
        conv_out = self.conv(x)          # (batch, 1)
        conv_out = conv_out.view(-1, 1)   # (batch, 1)
        features = self.regressor(conv_out)  # (batch, 1)
        return features
