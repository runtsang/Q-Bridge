"""Hybrid classical regression model with residual fusion and feature‑wise attention."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# ------------------------------------------------------------
# Data generation
# ------------------------------------------------------------
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Produce a synthetic dataset where each sample is a vector of real numbers
    and the target is a smooth sinusoidal function of the sum of the
    components, perturbed by a small cosine term.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

# ------------------------------------------------------------
# Dataset wrapper
# ------------------------------------------------------------
class RegressionDataset(Dataset):
    """
    Returns a dictionary with raw feature vectors and corresponding targets.
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

# ------------------------------------------------------------
# Classical regression network
# ------------------------------------------------------------
class ResidualMLP(nn.Module):
    """
    Small MLP with a residual connection that feeds into a linear head.
    """
    def __init__(self, num_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_features),
        )
        self.head = nn.Linear(num_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        resid = self.net(x)
        return self.head(resid + x)

class QModel(nn.Module):
    """
    Classical estimator that learns a feature‑wise attention mask before
    feeding the scaled features into a residual MLP.
    """
    def __init__(self, num_features: int):
        super().__init__()
        self.attn = nn.Parameter(torch.ones(num_features))
        self.residual = ResidualMLP(num_features)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        attn_scale = torch.sigmoid(self.attn)
        scaled = state_batch * attn_scale
        return self.residual(scaled)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
