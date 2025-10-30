"""Hybrid classical regression model inspired by Quantum‑NAT and the regression seed."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic regression data from superposition states.
    The target is a smooth function of the summed angles.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """
    Dataset that yields a feature vector and a scalar target.
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


class RegressionCNN(nn.Module):
    """
    1‑D convolutional feature extractor followed by a fully‑connected head.
    Mirrors the QFCModel architecture from Quantum‑NAT but operates on flat vectors.
    """
    def __init__(self, in_features: int):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        flattened = 16 * (in_features // 4)
        self.fc = nn.Sequential(
            nn.Linear(flattened, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.norm = nn.BatchNorm1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, features)
        x = x.unsqueeze(1)  # (batch, 1, features)
        feats = self.feature_extractor(x)
        flat = feats.view(feats.size(0), -1)
        out = self.fc(flat)
        return self.norm(out).squeeze(-1)


class QModel(nn.Module):
    """
    Classical regression model that applies a CNN feature extractor
    followed by a normalised linear head.
    """
    def __init__(self, num_features: int):
        super().__init__()
        self.net = RegressionCNN(num_features)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(state_batch)


__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
