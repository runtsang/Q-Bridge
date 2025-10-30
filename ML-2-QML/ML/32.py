"""Classical regression model with configurable depth and multi‑output support."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def generate_superposition_data(
    num_features: int,
    samples: int,
    *,
    noise: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data that mimics the quantum superposition
    used in the original example.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input feature vector.
    samples : int
        Number of samples to generate.
    noise : float, optional
        Standard deviation of Gaussian noise added to the target.

    Returns
    -------
    features : np.ndarray
        Shape (samples, num_features).
    targets : np.ndarray
        Shape (samples,).
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    if noise > 0.0:
        y += np.random.normal(scale=noise, size=y.shape)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset that yields a dict with ``states`` and ``target``."""

    def __init__(self, samples: int, num_features: int, noise: float = 0.0):
        self.features, self.labels = generate_superposition_data(num_features, samples, noise=noise)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QuantumRegression__gen033(nn.Module):
    """
    Classical feed‑forward regression model with flexible depth.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input.
    hidden_layers : list[int] | tuple[int], optional
        Sizes of hidden layers. Default is ``[32, 16]``.
    output_dim : int, optional
        Number of regression outputs. Default is 1.
    """

    def __init__(
        self,
        num_features: int,
        hidden_layers: list[int] | tuple[int] = (32, 16),
        output_dim: int = 1,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_features = num_features
        for h in hidden_layers:
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.ReLU())
            in_features = h
        layers.append(nn.Linear(in_features, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, num_features)``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, output_dim)``.
        """
        return self.net(x.to(torch.float32))

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convenience wrapper that detaches the output and returns a NumPy array.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, num_features)``.

        Returns
        -------
        np.ndarray
            Shape ``(batch, output_dim)``.
        """
        with torch.no_grad():
            return self.forward(x).cpu().numpy()
