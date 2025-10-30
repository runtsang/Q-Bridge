"""Hybrid classical regression model combining convolutional feature extraction and a fixed random layer.

The model processes 1‑D feature vectors with a lightweight CNN, applies a frozen random linear
transformation (mimicking the RandomLayer in the quantum version), and finally predicts a scalar
regression target.  This mirrors the quantum encoder/measurement pipeline while staying fully
classical."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic regression data where the target is a noisy sinusoid of the sum of features.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """
    Dataset wrapping the synthetic regression data.
    """
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class HybridRegressionModel(nn.Module):
    """
    Classical regression model that emulates the quantum encoder/measurement pipeline.
    """
    def __init__(self, num_features: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        # Determine the flattened feature size after two poolings
        flattened_size = 16 * (num_features // 4)
        self.random_linear = nn.Linear(flattened_size, 128, bias=False)
        # Freeze random weights to mimic a non‑trainable RandomLayer
        self.random_linear.weight.requires_grad = False
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: reshape, extract features, apply random linear, and predict.
        """
        bsz = x.shape[0]
        # Reshape to (batch, channel=1, length)
        features = self.features(x.unsqueeze(1))
        flattened = features.view(bsz, -1)
        random_features = self.random_linear(flattened)
        out = self.fc(random_features)
        return out.squeeze(-1)


__all__ = ["HybridRegressionModel", "RegressionDataset", "generate_superposition_data"]
