"""Hybrid classical-quantum feature extractor with a learnable encoder.

The QFCModelV2 extends the original QFCModel by adding a deeper convolutional
encoder, a lightweight classifier head, and batch‑norm for stable gradients.
This structure allows the model to learn more expressive features before
passing them to a quantum module in the QML counterpart.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class QFCModelV2(nn.Module):
    """A hybrid classical‑quantum feature extractor with a learnable encoder.

    The network consists of:
    1. A convolutional encoder that extracts hierarchical features from a
       single‑channel image.
    2. A fully‑connected classifier head that maps the flattened features to
       four output logits.
    3. Batch‑normalisation layers to stabilise training.
    """

    def __init__(self) -> None:
        super().__init__()
        # Convolutional encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Calculate flattened size: input 28x28 -> after 3 poolings -> 3x3
        self.flatten_size = 64 * 3 * 3

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(64, 4),
        )
        # Batch‑norm on final logits
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Normalised logits of shape (batch, 4).
        """
        features = self.encoder(x)
        flattened = features.view(features.size(0), -1)
        logits = self.classifier(flattened)
        return self.norm(logits)

__all__ = ["QFCModelV2"]
