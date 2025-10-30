"""Hybrid classical regression model combining convolutional feature extraction with a variational head."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset producing 1‑D feature vectors and regression targets."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int):
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QModel(nn.Module):
    """
    Classical regression model that first extracts local patterns with a lightweight CNN
    (mirroring the Quantum‑NAT feature extractor) and then predicts the target with a
    fully‑connected “variational” head.  The architecture is deliberately simple so
    that it can be used as a baseline against the quantum counterpart.
    """
    def __init__(self, num_features: int = 32, conv_channels: int = 8):
        super().__init__()
        # Conv1d treats each feature vector as a 1‑D image.
        self.features = nn.Sequential(
            nn.Conv1d(1, conv_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(conv_channels, conv_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        # Compute the flattened size after pooling.
        dummy = torch.zeros(1, 1, num_features)
        flat_size = self.features(dummy).view(1, -1).shape[1]
        self.fc = nn.Sequential(
            nn.Linear(flat_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.norm = nn.BatchNorm1d(1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        state_batch : torch.Tensor
            Shape (B, N) where N is the number of features.
        Returns
        -------
        torch.Tensor
            Shape (B,) regression predictions.
        """
        # Reshape to (B, 1, N) for Conv1d.
        x = state_batch.unsqueeze(1)
        feats = self.features(x)
        flat = feats.view(state_batch.shape[0], -1)
        out = self.fc(flat).squeeze(-1)
        return self.norm(out)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
