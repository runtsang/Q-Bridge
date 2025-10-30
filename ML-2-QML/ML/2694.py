"""Hybrid classical regression model with optional fully connected surrogate.

The module defines a classical regression dataset, a surrogate fully‑connected layer (FCL),
and a composite model that can optionally incorporate the surrogate.  The API mirrors
the original QuantumRegression.py while extending it with a modular FCL component.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Iterable

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data that mimics superposition‑style labels."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Classic regression dataset."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class FCL(nn.Module):
    """Classical surrogate for a fully‑connected quantum layer."""
    def __init__(self, n_features: int = 1):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.linear(values)).mean(dim=0)
        return expectation.detach().numpy()

class QuantumRegressionModel(nn.Module):
    """Classical regression model that can wrap a classical FCL."""
    def __init__(
        self,
        num_features: int,
        hidden_sizes: Iterable[int] = (32, 16),
        use_fcl: bool = False,
    ):
        super().__init__()
        layers = []
        in_dim = num_features
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        self.use_fcl = use_fcl
        if use_fcl:
            self.fcl = FCL(num_features)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        out = self.net(state_batch.to(torch.float32)).squeeze(-1)
        if self.use_fcl:
            theta_seq = out.detach().cpu().numpy()
            out = torch.tensor(self.fcl.run(theta_seq), dtype=torch.float32)
        return out

__all__ = ["QuantumRegressionModel", "RegressionDataset", "generate_superposition_data"]
