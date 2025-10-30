"""Hybrid classical ConvGen124 module.

This module implements a classical convolutional filter followed by a
fully‑connected projection to four outputs.  It is drop‑in compatible
with the original Conv.py but adds a learnable batch‑norm head and
optionally accepts a callable quantum feature extractor that can be
plugged in for hybrid experiments.

The architecture:
    1. Conv2d(1, 8, kernel_size=3, padding=1)
    2. ReLU → MaxPool2d(2)
    3. Conv2d(8, 16, kernel_size=3, padding=1)
    4. ReLU → MaxPool2d(2)
    5. Flatten → Linear(16*7*7, 64) → ReLU → Linear(64, 4)
    6. BatchNorm1d(4)

The forward method accepts a batch of grayscale images of shape
(batch, 1, 28, 28).  If a quantum extractor is supplied, its output
is added element‑wise to the classical logits before batch‑norm.
"""
import torch
from torch import nn
from typing import Callable, Optional

class ConvGen124(nn.Module):
    """
    Classical convolutional backbone with optional quantum augmentation.
    """

    def __init__(self, quantum_extractor: Optional[Callable] = None):
        super().__init__()
        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Fully‑connected head
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)

        self.quantum_extractor = quantum_extractor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Output logits of shape (batch, 4).
        """
        # Classical feature extraction
        features = self.features(x)
        flattened = features.view(features.size(0), -1)
        logits = self.fc(flattened)

        # Optional quantum augmentation
        if self.quantum_extractor is not None:
            # Expected to return a tensor of shape (batch, 4)
            quantum_logits = self.quantum_extractor(x)
            logits = logits + quantum_logits

        return self.norm(logits)

__all__ = ["ConvGen124"]
