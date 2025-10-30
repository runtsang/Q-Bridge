"""Classical regression model with configurable architecture and dropout."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def generate_superposition_data(num_features: int, samples: int, noise_std: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic regression data.

    Parameters
    ----------
    num_features : int
        Number of input features.
    samples : int
        Number of data points.
    noise_std : float, optional
        Standard deviation of additive Gaussian noise on the labels.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    if noise_std > 0.0:
        y += np.random.normal(scale=noise_std, size=y.shape).astype(np.float32)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """
    Dataset wrapper for the synthetic regression data.
    """

    def __init__(self, samples: int, num_features: int, noise_std: float = 0.0):
        self.features, self.labels = generate_superposition_data(num_features, samples, noise_std)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class RegressionModel(nn.Module):
    """
    Flexible feed‑forward network for regression.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input.
    hidden_dims : list[int], optional
        Sizes of hidden layers. Default is [64, 32].
    activation : nn.Module, optional
        Non‑linearity applied after each hidden layer. Default is nn.ReLU().
    dropout : float, optional
        Dropout probability applied after each hidden layer. Default is 0.1.
    residual : bool, optional
        If True, adds residual connections between consecutive hidden layers.
    """

    def __init__(
        self,
        num_features: int,
        hidden_dims: list[int] | None = None,
        activation: nn.Module | None = None,
        dropout: float = 0.1,
        residual: bool = False,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [64, 32]
        activation = activation or nn.ReLU()
        layers = []
        in_dim = num_features
        for out_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(activation)
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            if residual and in_dim == out_dim:
                # residual connection will be added in forward
                pass
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        self.residual = residual

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = state_batch
        idx = 0
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                prev = x
                x = layer(x)
                if self.residual and prev.shape == x.shape:
                    x = x + prev
            else:
                x = layer(x)
        return x.squeeze(-1)


__all__ = ["RegressionModel", "RegressionDataset", "generate_superposition_data"]
