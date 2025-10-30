"""QuantumRegressionHybrid: classical regression backbone with optional quantum head.

This module implements a lightweight PyTorch regression model that can be
used independently or coupled with a quantum backend.  The design follows
the classical seed but adds a flexible interface for plugging a quantum
module.

Classes
-------
RegressionDataset
    Dataset that can return either real‑valued feature vectors or complex
    state vectors depending on *use_complex*.
QModel
    Purely classical feed‑forward network that mirrors the structure of the
    quantum seed but without any quantum operations.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int,
                                samples: int,
                                *,
                                use_complex: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate data that mimics a quantum superposition of 0/1 states.
    Parameters
    ----------
    num_features : int
        Number of real‑valued features or qubit count.
    samples : int
        Number of samples to generate.
    use_complex : bool, default False
        If True, produce a complex vector for each sample; otherwise
        produce a single real value per sample.
    Returns
    -------
    x, y : ndarray
        2‑D arrays of shape (samples, num_features) and (samples,).
    """
    if not use_complex:
        x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
        angles = x.sum(axis=1)
        y = np.sin(angles) + 0.1 * np.cos(2 * angles)
        return x, y.astype(np.float32)

    # Complex state vectors
    omega_0 = np.zeros(2 ** num_features, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_features, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_features), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels.astype(np.float32)

class RegressionDataset(Dataset):
    """
    Dataset that returns either real‑valued features or complex quantum
    states, together with a continuous target.
    """
    def __init__(self, samples: int, num_features: int, *, use_complex: bool = False):
        self.states, self.labels = generate_superposition_data(num_features, samples, use_complex=use_complex)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat if isinstance(self.states[idx], complex) else torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class QModel(nn.Module):
    """
    Purely classical regression network.  It matches the depth of the
    quantum seed but contains only standard PyTorch layers, making it
    fast and suitable for baseline experiments.
    """
    def __init__(self, num_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(state_batch).squeeze(-1)

__all__ = ["RegressionDataset", "QModel", "generate_superposition_data"]
