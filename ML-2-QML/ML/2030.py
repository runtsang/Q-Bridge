"""Classical regression dataset and model with enhanced architecture.

This module defines :class:`QuantumRegression` which is a deep residual network
with batch‑normalisation and dropout, trained on a synthetic dataset produced by
``generate_regression_data``.  The dataset is a toy “super‑position” signal
but with an additional Gaussian bump to give the model a harder learning
objective.  The class is fully torch‑compatible and can be dropped into any
PyTorch training script.

Author: GPT‑OSS‑20B
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def generate_regression_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate features ``x`` and labels ``y`` with a more complex signal.

    The features are uniformly sampled from ``[-1, 1]``.  The labels are
    ``sin(∑x) + 0.1*cos(2∑x) + 0.05*exp(-∑x²)`` giving a mixture of
    periodic and localised behaviour that encourages the network to learn
    non‑linear interactions.

    Parameters
    ----------
    num_features : int
        Dimensionality of each sample.
    samples : int
        Number of samples to generate.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``features`` of shape ``(samples, num_features)`` and ``labels`` of
        shape ``(samples,)``.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = (
        np.sin(angles)
        + 0.1 * np.cos(2 * angles)
        + 0.05 * np.exp(-np.square(angles))
    )
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """
    Torch ``Dataset`` that returns a dictionary with keys ``states`` and
    ``target`` for use in a ``DataLoader``.
    """

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_regression_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class ResidualBlock(nn.Module):
    """
    A simple residual block with two linear layers, batch‑norm and ReLU.
    """

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.relu(x + self.net(x))


class QuantumRegression(nn.Module):
    """
    A deep residual network tailored for the synthetic regression task.

    The architecture consists of an initial linear layer, followed by a
    configurable number of :class:`ResidualBlock` modules and a final linear
    head.  Dropout and batch‑normalisation help regularise the model when the
    synthetic data contains both global and local patterns.
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 64,
        num_blocks: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.initial = nn.Linear(num_features, hidden_dim)
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        h = nn.functional.relu(self.initial(state_batch))
        for block in self.blocks:
            h = block(h)
        return self.head(h).squeeze(-1)


__all__ = ["QuantumRegression", "RegressionDataset", "generate_regression_data"]
