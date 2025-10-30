"""Establishes a classical regression model that mirrors the quantum workflow.

The module provides:
- A deterministic data generator that produces superposition‑style inputs.
- A PyTorch Dataset compatible with the generated data.
- An MLP (EstimatorQNN__gen501) that uses batch‑norm, dropout, and ReLU.
- A convenient ``predict`` helper for inference on new tensors.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

__all__ = ["EstimatorQNN__gen501", "RegressionDataset", "generate_superposition_data"]


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Produce real‑valued feature vectors and target labels that emulate a
    superposition of two orthogonal states.  The function is intentionally
    deterministic enough to serve as a toy regression benchmark while
    preserving a non‑trivial, non‑linear relationship.

    Parameters
    ----------
    num_features : int
        Number of input features.
    samples : int
        Number of examples to generate.

    Returns
    -------
    x : np.ndarray, shape (samples, num_features)
        Feature matrix.
    y : np.ndarray, shape (samples,)
        Target values.
    """
    rng = np.random.default_rng()
    x = rng.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """PyTorch Dataset wrapping the synthetic superposition data."""

    def __init__(self, samples: int, num_features: int = 2):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class EstimatorQNN__gen501(nn.Module):
    """
    A depth‑3 MLP that accepts 2‑dimensional inputs (extendable via ``num_features``).

    The network architecture:
        Linear → BatchNorm → ReLU → Dropout
        Linear → BatchNorm → ReLU → Dropout
        Linear → Linear

    The final layer is linear to produce a scalar regression output.
    """

    def __init__(self, num_features: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x).squeeze(-1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience helper that runs a forward pass without gradient tracking."""
        with torch.no_grad():
            return self(x).cpu()
