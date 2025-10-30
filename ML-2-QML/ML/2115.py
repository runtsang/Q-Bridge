"""Enhanced classical regression module with residual network and feature augmentation.

The model is designed to mirror the quantum benchmark while providing
additional expressive power:
* The dataset includes polynomial and interaction terms to emulate
  higher‑order quantum correlations.
* The neural network is a deep residual network with dropout and
  layer normalisation, improving convergence on noisy data.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from itertools import combinations_with_replacement


def generate_superposition_data(
    num_features: int,
    samples: int,
    noise_std: float = 0.05,
    poly_degree: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a synthetic regression dataset.

    Parameters
    ----------
    num_features:
        Number of input features.
    samples:
        Number of data points to generate.
    noise_std:
        Standard deviation of Gaussian noise added to the target.
    poly_degree:
        Maximum degree of polynomial interaction terms to append to the raw
        features.  The function expands the feature vector to include all
        unique monomials up to ``poly_degree``.
    """
    # Base features drawn from a uniform distribution.
    base = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)

    # Construct polynomial interaction terms.
    if poly_degree > 1:
        poly_terms = []
        for deg in range(2, poly_degree + 1):
            for idxs in combinations_with_replacement(range(num_features), deg):
                term = np.prod(base[:, idxs], axis=1, keepdims=True)
                poly_terms.append(term)
        poly_features = np.hstack(poly_terms)
        features = np.hstack([base, poly_features])
    else:
        features = base

    # Target is a smooth non‑linear function of the angles.
    angles = features.sum(axis=1)
    labels = np.sin(angles) + 0.1 * np.cos(2 * angles)

    # Add Gaussian noise.
    labels += np.random.normal(0.0, noise_std, size=labels.shape)

    return features.astype(np.float32), labels.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset returning ``states`` and ``target`` tensors."""

    def __init__(self, samples: int, num_features: int, **kwargs):
        self.features, self.labels = generate_superposition_data(num_features, samples, **kwargs)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class ResidualBlock(nn.Module):
    """Simple residual block with two linear layers and skip connection."""

    def __init__(self, dim: int):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layer(x)


class QModel(nn.Module):
    """Deep residual neural network for regression."""

    def __init__(self, num_features: int, hidden_dim: int = 64, depth: int = 5):
        super().__init__()
        self.input_proj = nn.Linear(num_features, hidden_dim)
        self.residuals = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(depth)])
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.input_proj(state_batch)
        for block in self.residuals:
            x = block(x)
        return self.output_layer(x).squeeze(-1)


__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
