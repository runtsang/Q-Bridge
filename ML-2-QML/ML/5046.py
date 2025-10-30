"""Hybrid regression model with classical random‑feature expansion.

The model is designed to mirror the quantum version while remaining fully
classical.  It uses a clipped linear layer (borrowed from the fraud‑detection
seed) as a feature extractor, followed by a standard feed‑forward head.  The
dataset generator is identical to the quantum seed so that both models
operate on the same data distribution.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate data from a superposition of |0...0> and |1...1>.

    Parameters
    ----------
    num_features : int
        Number of input dimensions (features).
    samples : int
        Number of data points to generate.

    Returns
    -------
    X : np.ndarray
        Input features of shape (samples, num_features).
    y : np.ndarray
        Regression targets.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset wrapper around ``generate_superposition_data``."""

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class ClippedLinear(nn.Module):
    """Linear layer with optional clipping of weights and bias.

    Inspired by the fraud‑detection ``_layer_from_params`` helper.
    """

    def __init__(self, in_features: int, out_features: int, clip: bool = False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        if clip:
            with torch.no_grad():
                self.linear.weight.clamp_(-5.0, 5.0)
                self.linear.bias.clamp_(-5.0, 5.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.linear(x)


class QuantumInspiredFeatureLayer(nn.Module):
    """Deterministic random Fourier feature map.

    The mapping matrix is sampled once at construction and kept fixed.
    This mimics the unitary encoding of the quantum circuit.
    """

    def __init__(self, in_features: int, out_features: int, seed: int = 42):
        super().__init__()
        rng = np.random.default_rng(seed)
        # Random matrix with entries in [-pi, pi]
        self.W = nn.Parameter(
            torch.tensor(rng.uniform(-np.pi, np.pi, size=(out_features, in_features)), dtype=torch.float32),
            requires_grad=False,
        )
        self.b = nn.Parameter(
            torch.tensor(rng.uniform(0, 2 * np.pi, size=(out_features,)), dtype=torch.float32),
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # x: (batch, in_features)
        y = torch.matmul(x, self.W.t()) + self.b
        return torch.cos(y)  # cosine kernel


class HybridRegressionModel(nn.Module):
    """Classical regression head that mirrors the quantum variant.

    The architecture consists of:
    1. A random Fourier feature layer (quantum‑inspired).
    2. A clipped linear layer (from fraud detection).
    3. A small feed‑forward head with ReLU activations.
    4. Optional batch‑norm on the final output.
    """

    def __init__(self, num_features: int, feature_dim: int = 64):
        super().__init__()
        self.feature_extractor = QuantumInspiredFeatureLayer(num_features, feature_dim)
        self.clipped = ClippedLinear(feature_dim, feature_dim // 2, clip=True)
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
            nn.BatchNorm1d(1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_extractor(state_batch)
        x = self.clipped(x)
        out = self.head(x)
        return out.squeeze(-1)


__all__ = ["HybridRegressionModel", "RegressionDataset", "generate_superposition_data"]
