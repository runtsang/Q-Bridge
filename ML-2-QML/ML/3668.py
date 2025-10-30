"""Hybrid regression model combining classical MLP and a shallow estimator network.

The module extends the original QuantumRegression.py by adding a two-branch
architecture: a deep MLP and a lightweight estimator branch inspired by
EstimatorQNN.  The two branches are concatenated before the final linear
regression head.  This design allows experiments with different
classical inductive biases while keeping the dataset generation
compatible with the quantum counterpart.

The dataset generation and utilities are preserved for consistency with
the original seed, but the class names have been updated to reflect the
hybrid nature.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data by summing sinusoidal features.

    Parameters
    ----------
    num_features : int
        Number of input features.
    samples : int
        Number of samples to generate.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Features of shape (samples, num_features) and targets of shape
        (samples,).
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset wrapping the synthetic regression data."""

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class EstimatorNN(nn.Module):
    """A lightweight feed‑forward regressor inspired by EstimatorQNN."""

    def __init__(self, in_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(inputs)


class DeepNN(nn.Module):
    """A deeper MLP mirroring the original QModel implementation."""

    def __init__(self, in_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(state_batch.to(torch.float32)).squeeze(-1)


class HybridRegressionModel(nn.Module):
    """Two‑branch classifier combining EstimatorNN and DeepNN."""

    def __init__(self, num_features: int):
        super().__init__()
        self.estimator = EstimatorNN(num_features)
        self.deep = DeepNN(num_features)
        self.head = nn.Linear(2, 1)  # concatenated outputs from both branches

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """Forward pass producing a scalar regression prediction."""
        est_out = self.estimator(state_batch)
        deep_out = self.deep(state_batch)
        concat = torch.cat([est_out.unsqueeze(-1), deep_out.unsqueeze(-1)], dim=-1)
        return self.head(concat).squeeze(-1)


__all__ = ["HybridRegressionModel", "RegressionDataset", "generate_superposition_data"]
