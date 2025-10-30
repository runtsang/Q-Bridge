"""Hybrid sampler‑regressor combining classical SamplerQNN and quantum regression ideas.

The module defines:
- SamplerQNNGen048: a neural network that outputs a 2‑class probability distribution and a scalar regression value.
- RegressionDataset: dataset for generating superposition states and labels.
- generate_superposition_data: helper to produce synthetic data.
- SamplerQNN: factory function for backward compatibility.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

__all__ = ["SamplerQNNGen048", "RegressionDataset", "generate_superposition_data", "SamplerQNN"]


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data: states in R^num_features and target values."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset that returns states and regression targets."""

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class SamplerQNNGen048(nn.Module):
    """
    Hybrid sampler‑regressor.

    The network first maps the input through a small feed‑forward encoder,
    then splits into two heads:
    * sampler_head: outputs logits for a 2‑class softmax distribution.
    * regression_head: outputs a scalar regression value.
    """

    def __init__(self, num_features: int = 2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_features, 4),
            nn.Tanh(),
            nn.Linear(4, 4),
            nn.Tanh(),
        )
        self.sampler_head = nn.Linear(4, 2)
        self.regression_head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        x = self.encoder(inputs)
        sampler_logits = self.sampler_head(x)
        distribution = F.softmax(sampler_logits, dim=-1)
        regression = self.regression_head(x).squeeze(-1)
        return distribution, regression


def SamplerQNN() -> SamplerQNNGen048:
    """Factory function for backward compatibility with legacy code."""
    return SamplerQNNGen048()
