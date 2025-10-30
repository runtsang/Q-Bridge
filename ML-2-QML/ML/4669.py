"""Hybrid regression model that optionally delegates to a quantum module.

The classical branch implements a standard MLP.  The quantum branch is
expected to be supplied externally; the class is designed to be drop‑in
compatible with the original ``QuantumRegression`` API.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data using a sinusoidal target."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset that mirrors the quantum example but stores tensors."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class HybridRegression(nn.Module):
    """Drop‑in replacement that can operate in classical or quantum mode."""
    def __init__(self, num_features: int, hidden_dim: int = 32, use_quantum: bool = False):
        super().__init__()
        self.use_quantum = use_quantum
        if use_quantum:
            # placeholder; the user should assign a quantum module later
            self.quantum_head = nn.Identity()
        else:
            self.net = nn.Sequential(
                nn.Linear(num_features, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            )

    def set_quantum_module(self, qm: nn.Module) -> None:
        """Attach a quantum module that produces a single‑dimensional output."""
        self.quantum_head = qm
        self.use_quantum = True

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        if self.use_quantum:
            return self.quantum_head(states).squeeze(-1)
        return self.net(states).squeeze(-1)


__all__ = ["HybridRegression", "RegressionDataset", "generate_superposition_data"]
