"""Classical regression model with quantum‑inspired random Fourier features.

The module defines ``QuantumRegression__gen334`` that transforms input features
via a fixed random Fourier map before applying a linear regression head.
It also contains utilities for data generation and a PyTorch dataset.
"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic regression dataset where the target is a nonlinear
    function of a sum of the input features. The function mirrors the
    structure used in the original seed but exposes a seed argument
    for reproducibility.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input vectors.
    samples : int
        Number of samples to generate.
    seed : int | None
        Optional random seed.

    Returns
    -------
    X : np.ndarray of shape (samples, num_features)
        Feature matrix.
    y : np.ndarray of shape (samples,)
        Continuous target values.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = X.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return X, y.astype(np.float32)


class RegressionDataset(Dataset):
    """
    PyTorch dataset that yields a dictionary containing the features
    under the key ``states`` and the target under ``target``.
    """
    def __init__(self, samples: int, num_features: int, seed: int | None = None):
        self.features, self.labels = generate_superposition_data(num_features, samples, seed)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class QuantumRegression__gen334(nn.Module):
    """
    Classical regression model that uses a random Fourier feature map
    to emulate a “quantum” feature space.  The model consists of:
    * A fixed random Fourier feature layer (no trainable parameters).
    * A linear head that maps the expanded features to a scalar output.
    """
    def __init__(self, num_features: int, feature_dim: int = 256, device: torch.device | str | None = None):
        super().__init__()
        self.num_features = num_features
        self.feature_dim = feature_dim
        self.device = torch.device(device) if device is not None else torch.device("cpu")

        # Random Fourier parameters – these are fixed after initialization.
        rng = np.random.default_rng()
        self.register_buffer("W", torch.tensor(rng.normal(size=(num_features, feature_dim)), dtype=torch.float32, device=self.device))
        self.register_buffer("b", torch.tensor(rng.uniform(0, 2 * np.pi, size=(feature_dim,)), dtype=torch.float32, device=self.device))
        self.scaling = np.sqrt(2.0 / feature_dim)

        # Linear head
        self.head = nn.Linear(feature_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor of shape (batch, num_features)

        Returns
        -------
        output : torch.Tensor of shape (batch,)
        """
        # Random Fourier feature map
        z = torch.cos(x @ self.W + self.b) * self.scaling
        # Linear regression head
        return self.head(z).squeeze(-1)


__all__ = ["QuantumRegression__gen334", "RegressionDataset", "generate_superposition_data"]
