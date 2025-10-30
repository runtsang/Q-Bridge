"""Hybrid classical regression model with optional quantum‑style embeddings.

The module implements a fully‑connected neural network that can be
used as a drop‑in replacement for the original FCL.  It also exposes
the same dataset utilities as the quantum counterpart so experiments
can be run side‑by‑side.
"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data.

    The target is a smooth sinusoidal function of the sum of the
    input features, with a small cosine perturbation to introduce
    non‑linearity.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """PyTorch dataset wrapping the generated superposition data."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class HybridRegressionLayer(nn.Module):
    """Classical fully‑connected regression network.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input feature vector.
    hidden_dim : int, default 64
        Width of the hidden layers.
    n_layers : int, default 3
        Number of linear layers (including the output layer).
    """
    def __init__(self, num_features: int, hidden_dim: int = 64, n_layers: int = 3) -> None:
        super().__init__()
        layers = []
        in_dim = num_features
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """Return a regression score for each batch element."""
        return self.net(state_batch).squeeze(-1)

def FCL() -> HybridRegressionLayer:
    """Factory that returns a default instance."""
    return HybridRegressionLayer(num_features=1)

__all__ = ["HybridRegressionLayer", "generate_superposition_data", "RegressionDataset", "FCL"]
