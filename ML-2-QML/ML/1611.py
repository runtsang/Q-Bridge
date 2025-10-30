"""Enhanced classical regression model with dropout and batch‑norm."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample synthetic data that mimics superposition‑like behaviour.

    Parameters
    ----------
    num_features : int
        Number of input features.
    samples : int
        Number of samples to generate.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Feature matrix and target vector.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """
    PyTorch dataset that returns state tensors and targets.
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


class QuantumRegressionGen177(nn.Module):
    """
    Feed‑forward network with optional residual connections and dropout.
    """

    def __init__(
        self,
        num_features: int,
        hidden_sizes: list[int] | tuple[int,...] = (64, 32),
        dropout: float = 0.1,
        use_residual: bool = False,
    ):
        """
        Parameters
        ----------
        num_features : int
            Input dimensionality.
        hidden_sizes : list or tuple of int, optional
            Sizes of hidden layers. Default (64, 32).
        dropout : float, optional
            Dropout probability. Default 0.1.
        use_residual : bool, optional
            Whether to add skip connections. Default False.
        """
        super().__init__()
        layers: list[nn.Module] = []
        in_features = num_features

        for h in hidden_sizes:
            layers += [
                nn.Linear(in_features, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ]
            in_features = h

        layers.append(nn.Linear(in_features, 1))
        self.net = nn.Sequential(*layers)
        self.use_residual = use_residual
        if use_residual and num_features!= hidden_sizes[-1]:
            raise ValueError("Residual path must match feature dimension.")

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        state_batch : torch.Tensor
            Batch of input states, shape (batch_size, num_features).

        Returns
        -------
        torch.Tensor
            Predicted values, shape (batch_size,).
        """
        out = self.net(state_batch.to(torch.float32))
        if self.use_residual:
            out += state_batch
        return out.squeeze(-1)


__all__ = ["QuantumRegressionGen177", "RegressionDataset", "generate_superposition_data"]
