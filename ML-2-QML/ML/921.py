"""Advanced classical regression model and dataset.

This module extends the original seed by adding data‑generation flexibility,
a residual neural network and a lightweight training utility.  The API remains
compatible with the original names (`RegressionDataset`, `generate_regression_data`,
`QuantumRegressionModel`), but the implementation is richer and ready for
research experiments.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def generate_regression_data(num_features: int, samples: int,
                             dist: str = "uniform", noise: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data.

    Parameters
    ----------
    num_features: int
        Dimensionality of the input space.
    samples: int
        Number of samples to generate.
    dist: str, optional
        Distribution of the features: ``"uniform"`` or ``"normal"``.
    noise: float, optional
        Standard deviation of Gaussian noise added to the target.
    """
    rng = np.random.default_rng()
    if dist == "uniform":
        X = rng.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    elif dist == "normal":
        X = rng.standard_normal(size=(samples, num_features)).astype(np.float32)
    else:
        raise ValueError(f"Unsupported distribution {dist!r}")

    angles = X.sum(axis=1)
    y = np.sin(angles) + noise * rng.standard_normal(samples)
    return X, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset that wraps the synthetic data generator."""

    def __init__(self, samples: int, num_features: int,
                 dist: str = "uniform", noise: float = 0.1):
        self.features, self.labels = generate_regression_data(
            num_features, samples, dist=dist, noise=noise
        )

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class QuantumRegressionModel(nn.Module):
    """Residual neural network for regression."""

    def __init__(self, num_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.residual = nn.Linear(num_features, 64)
        self.head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        out += self.residual(x)
        return self.head(out).squeeze(-1)

__all__ = ["QuantumRegressionModel", "RegressionDataset", "generate_regression_data"]
