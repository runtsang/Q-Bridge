"""Classical convolution‑inspired network with modern regularisation."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["QCNNModel"]


class QCNNModel(nn.Module):
    """
    A deeper, regularised version of the original QCNN‑style network.

    The model consists of:
    * A feature extractor that expands the 8‑dimensional input to a 32‑dimensional space.
    * Three convolution‑like blocks (Linear → BatchNorm → ReLU → Dropout).
    * A 3‑layer fully‑connected head that maps the 8‑dimensional bottleneck to a single sigmoid output.

    The design mirrors the quantum convolution layers but adds modern best practices
    such as batch‑normalisation and dropout for improved generalisation.
    """

    def __init__(self) -> None:
        super().__init__()

        # Feature map: 8 → 32
        self.feature_map = nn.Sequential(
            nn.Linear(8, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        )

        # Convolution‑like blocks
        self.conv1 = nn.Sequential(
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        )
        self.pool1 = nn.Sequential(
            nn.Linear(32, 24),
            nn.BatchNorm1d(24),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Linear(24, 24),
            nn.BatchNorm1d(24),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        )
        self.pool2 = nn.Sequential(
            nn.Linear(24, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Linear(16, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        )
        self.pool3 = nn.Sequential(
            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        )

        # Classical head
        self.head = nn.Sequential(
            nn.Linear(8, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        logits = self.head(x)
        return torch.sigmoid(logits)
