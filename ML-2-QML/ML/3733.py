"""Hybrid classical sampler and regression head.

The class HybridSamplerRegressor produces:
- a probability distribution over two input angles (classical sampler)
- four trainable weight angles for a variational quantum circuit
- a lightweight regression output for the same input.

The module also provides a synthetic dataset generator identical to the
reference regression data, but enriched with a probability head.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class HybridSamplerRegressor(nn.Module):
    """Classical sampler + regression head."""
    def __init__(self) -> None:
        super().__init__()
        # Encoder: raw features -> two input angles
        self.input_encoder = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )
        # Weight generator: raw features -> 4 variational angles
        self.weight_gen = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
        )
        # Regression head for auxiliary task
        self.reg_head = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return a dictionary with classical probabilities, weight params
        and a regression output."""
        probs = F.softmax(self.input_encoder(x), dim=-1)
        weights = self.weight_gen(x)
        reg = self.reg_head(x).squeeze(-1)
        return {"probs": probs, "weights": weights, "reg": reg}

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data resembling a quantum superposition."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset yielding both classical features and target."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        return {
            "features": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

def SamplerQNN() -> HybridSamplerRegressor:
    """Convenience wrapper matching the original anchor."""
    return HybridSamplerRegressor()

__all__ = ["HybridSamplerRegressor", "RegressionDataset",
           "generate_superposition_data", "SamplerQNN"]
