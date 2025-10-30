"""Hybrid classical QFCModel with classification and regression heads.

This module extends the original QuantumNAT convolutional network by adding a
regression head that can be invoked via `reg_forward`.  The class also
provides a static data‑generation helper for synthetic superposition data,
mirroring the quantum version.  The architecture is fully compatible with
PyTorch and can be plugged into existing pipelines that expect a
`nn.Module` with a `forward` method returning a 4‑dimensional output.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class QFCModel(nn.Module):
    """Hybrid CNN with optional regression head."""
    def __init__(self, num_reg_features: int = 64, reg_hidden: int = 32) -> None:
        super().__init__()
        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)
        # Regression head
        self.feature_extractor = nn.Sequential(
            nn.Linear(16 * 7 * 7, num_reg_features),
            nn.ReLU(),
        )
        self.reg_head = nn.Sequential(
            nn.Linear(num_reg_features, reg_hidden),
            nn.ReLU(),
            nn.Linear(reg_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return classification logits."""
        bsz = x.shape[0]
        features = self.features(x)
        flattened = features.view(bsz, -1)
        out = self.fc(flattened)
        return self.norm(out)

    def reg_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return regression prediction from the same input."""
        bsz = x.shape[0]
        features = self.features(x)
        flattened = features.view(bsz, -1)
        feat = self.feature_extractor(flattened)
        return self.reg_head(feat).squeeze(-1)

    @staticmethod
    def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
        """Generate synthetic data for regression."""
        x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
        angles = x.sum(axis=1)
        y = np.sin(angles) + 0.1 * np.cos(2 * angles)
        return x, y.astype(np.float32)

__all__ = ["QFCModel"]
