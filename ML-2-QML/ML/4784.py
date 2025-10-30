"""Hybrid classical implementation of a fully‑connected regression layer.

The module mirrors the behaviour of the original FCL, QuantumRegression and EstimatorQNN
seed projects.  It provides a PyTorch model that can be used in place of the quantum
circuit, and helper utilities for generating regression data and a PyTorch dataset.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


def generate_superposition_data(num_features: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic regression target from a superposition‑like pattern.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input feature vector.
    samples : int
        Number of samples to generate.

    Returns
    -------
    X, y : np.ndarray
        Features of shape (samples, num_features) and target of shape (samples,).
    """
    X = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = X.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return X, y.astype(np.float32)


class RegressionDataset(Dataset):
    """
    PyTorch wrapper around the synthetic regression data.

    The dataset yields a dict with ``states`` and ``target`` keys to match the
    expectations of downstream training loops.
    """

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class SharedClassName(nn.Module):
    """
    Classical fully‑connected regression network.

    Architecture
    ------------
    * Linear layer that maps ``n_features`` to a hidden size (32).
    * Two hidden layers with ReLU activations.
    * Final linear head producing a scalar output.
    """

    def __init__(self, n_features: int = 1, hidden_size: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(state_batch.to(torch.float32)).squeeze(-1)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Mimic the original ``FCL.run`` API.

        Parameters
        ----------
        thetas : Iterable[float]
            Iterable of parameters that will be injected into a linear layer.

        Returns
        -------
        np.ndarray
            The mean of the activation over the batch as a 1‑D array.
        """
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.net[0](values)).mean(dim=0)
        return expectation.detach().numpy()


__all__ = ["SharedClassName", "RegressionDataset", "generate_superposition_data"]
