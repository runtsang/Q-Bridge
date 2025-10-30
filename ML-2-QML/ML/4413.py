"""Hybrid classical regression model combining self‑attention and a fully‑connected layer.

This module extends the original `QuantumRegression.py` by adding a
self‑attention block and a fully‑connected layer that mimics the
parameterised quantum circuit from the QML seed.  The dataset generator
is kept identical so that the same synthetic data can be used for
benchmarking.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# ----------------------------------------------------------------------
# Data generation
# ----------------------------------------------------------------------
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic superposition data.

    The function is identical to the original seed but is re‑implemented
    here to keep the module self‑contained.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset that returns a dictionary with ``states`` and ``target``."""

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# ----------------------------------------------------------------------
# Classical self‑attention
# ----------------------------------------------------------------------
class ClassicalSelfAttention(nn.Module):
    """Simple dot‑product self‑attention block.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the query/key/value vectors.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        # Linear projections for query and key
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return attended representation."""
        q = self.q_proj(inputs)
        k = self.k_proj(inputs)
        v = inputs
        scores = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.embed_dim), dim=-1)
        return torch.matmul(scores, v)

# ----------------------------------------------------------------------
# Classical fully‑connected layer (mimicking a parameterised quantum circuit)
# ----------------------------------------------------------------------
class FullyConnectedLayer(nn.Module):
    """A tiny MLP that emulates the behaviour of a single‑qubit
    parameterised circuit from the QML seed.
    """

    def __init__(self, n_features: int = 1):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, thetas: torch.Tensor) -> torch.Tensor:
        """Return a scalar expectation value."""
        return torch.tanh(self.linear(thetas)).mean()

# ----------------------------------------------------------------------
# Hybrid regression model
# ----------------------------------------------------------------------
class QuantumRegressionHybrid(nn.Module):
    """Classical hybrid regression model.

    The architecture consists of:

    1. A linear encoder that projects the input into an
       ``embed_dim``‑dimensional space.
    2. A self‑attention block that captures pairwise interactions.
    3. A fully‑connected layer that acts as a quantum‑style
       parameterised circuit.
    4. A final linear head that outputs a scalar.
    """

    def __init__(self, num_features: int, embed_dim: int = 32):
        super().__init__()
        self.encoder = nn.Linear(num_features, embed_dim)
        self.attention = ClassicalSelfAttention(embed_dim)
        self.fcl = FullyConnectedLayer(embed_dim)
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """Forward pass through the hybrid pipeline."""
        encoded = self.encoder(state_batch)
        attended = self.attention(encoded)
        # Use the attended representation as the “theta” for the FCL
        fcl_out = self.fcl(attended)
        out = self.head(fcl_out)
        return out.squeeze(-1)

__all__ = ["QuantumRegressionHybrid", "RegressionDataset", "generate_superposition_data"]
