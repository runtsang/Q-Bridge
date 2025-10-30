"""Classical regression model and dataset with enhanced architecture.

The module provides:
- `generate_superposition_data` – creates synthetic data using a simple trigonometric
  mapping from input features.  A Gaussian noise term can be added to emulate
  measurement uncertainty.
- `RegressionDataset` – a torch Dataset that can optionally scale the features
  with a `StandardScaler`.
- `QuantumRegressionModel` – a residual neural network with dropout that
  supports arbitrary hidden layer sizes and activation choices.

The implementation is intentionally lightweight so it can be dropped into
existing PyTorch training pipelines.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


def generate_superposition_data(
    num_features: int,
    samples: int,
    noise_std: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic regression data.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input space.
    samples : int
        Number of samples to generate.
    noise_std : float, optional
        Standard deviation of Gaussian noise added to the labels.

    Returns
    -------
    X : ndarray of shape (samples, num_features)
        Input features drawn uniformly from [-1, 1].
    y : ndarray of shape (samples,)
        Target values computed as ``sin(x.sum(axis=1)) + 0.1*cos(2*x.sum(axis=1))``.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    if noise_std > 0.0:
        y += np.random.normal(scale=noise_std, size=y.shape).astype(np.float32)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """
    PyTorch ``Dataset`` that returns a feature tensor and a target scalar.

    Parameters
    ----------
    samples : int
        Number of samples in the dataset.
    num_features : int
        Dimensionality of the input space.
    scale_features : bool, default ``True``
        Whether to standardise the features to zero mean and unit variance.
    """

    def __init__(self, samples: int, num_features: int, scale_features: bool = True):
        self.features, self.labels = generate_superposition_data(num_features, samples)
        if scale_features:
            scaler = StandardScaler()
            self.features = scaler.fit_transform(self.features).astype(np.float32)
        else:
            self.features = self.features.astype(np.float32)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class QuantumRegressionModel(nn.Module):
    """
    Residual neural network for regression with optional dropout.

    Parameters
    ----------
    num_features : int
        Size of the input vector.
    hidden_sizes : Sequence[int], optional
        Sizes of the hidden layers.  Defaults to ``[32, 16]``.
    activation : nn.Module, optional
        Activation function to use after each hidden layer.
    dropout : float, optional
        Dropout probability applied after each hidden layer.
    """

    def __init__(
        self,
        num_features: int,
        hidden_sizes: tuple[int,...] = (32, 16),
        activation: nn.Module | None = nn.ReLU(),
        dropout: float = 0.0,
    ):
        super().__init__()
        layers = []
        in_size = num_features
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            if activation is not None:
                layers.append(activation)
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(h, in_size))  # residual connection
            layers.append(nn.ReLU())
            in_size = h
        layers.append(nn.Linear(in_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out = self.net(state_batch.to(torch.float32))
        return out.squeeze(-1)


__all__ = ["QuantumRegressionModel", "RegressionDataset", "generate_superposition_data"]
