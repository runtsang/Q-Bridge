"""Hybrid classical model combining CNN encoding, classification, and regression heads.

This module defines HybridNATModel which extends the original QuantumNAT classical
architecture by adding a regression head and a flexible input interface. It can handle
image data or pre‑encoded quantum states (as flattened vectors). The model outputs a
dictionary containing both classification logits and regression predictions.

Utilities for generating synthetic superposition data from the quantum regression
example are also provided for quick experimentation.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

__all__ = ["HybridNATModel", "RegressionDataset", "generate_superposition_data"]


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic superposition data for regression tasks."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset yielding pre‑computed superposition states and regression targets."""

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class HybridNATModel(nn.Module):
    """Hybrid classical model with CNN encoder, classification head, and optional regression head."""

    def __init__(self, n_classes: int = 4, regression: bool = True) -> None:
        super().__init__()
        # CNN encoder identical to original QuantumNAT
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Flattened feature dimension: 16 * 7 * 7
        self.classifier = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )
        self.class_norm = nn.BatchNorm1d(n_classes)

        self.regression = regression
        if regression:
            self.regressor = nn.Sequential(
                nn.Linear(16 * 7 * 7, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (B, 1, 28, 28).

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary with keys 'class' and optionally'regress'.
        """
        features = self.features(x)
        flattened = features.view(features.size(0), -1)
        class_logits = self.class_norm(self.classifier(flattened))

        outputs: dict[str, torch.Tensor] = {"class": class_logits}
        if self.regression:
            outputs["regress"] = self.regressor(flattened).squeeze(-1)
        return outputs
