"""Hybrid regression model combining classical encoding, a quantum-inspired fully connected layer, and a linear head."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def generate_hybrid_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic regression data with a mix of classical and quantum-like components.
    Classical features are sampled uniformly; quantum-like labels are derived from a
    superposition-inspired function to mimic the quantum seed.
    """
    # Classical feature matrix
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    # Quantum-inspired target using a trigonometric mixture
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class HybridDataset(Dataset):
    """
    Dataset wrapper that returns a dictionary with ``states`` and ``target``.
    ``states`` contain the classical feature vector; a quantum simulator could
    operate on these if needed.
    """
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_hybrid_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class QuantumInspiredFCL(nn.Module):
    """
    Classical stand‑in for a fully connected quantum layer.
    Implements a parameterized linear transform followed by a non‑linear squashing
    and a mean aggregation, mirroring the behaviour of the quantum FCL example.
    """
    def __init__(self, n_features: int = 1):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, thetas: torch.Tensor) -> torch.Tensor:
        # ``thetas`` are treated as a batch of parameters
        values = thetas.view(-1, 1).float()
        expectation = torch.tanh(self.linear(values)).mean(dim=0)
        return expectation


class HybridRegression(nn.Module):
    """
    Hybrid regression model consisting of:
    1. A classical linear encoder that projects raw features to a latent space.
    2. A quantum‑inspired fully connected layer (FCL) that aggregates
       the encoded features non‑linearly.
    3. A final linear head that maps the aggregated representation to a scalar output.
    """
    def __init__(self, num_features: int, latent_dim: int = 32):
        super().__init__()
        self.encoder = nn.Linear(num_features, latent_dim)
        self.fcl = QuantumInspiredFCL(latent_dim)
        self.head = nn.Linear(1, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        # Encode raw features
        encoded = torch.relu(self.encoder(state_batch))
        # Pass through quantum‑inspired FCL
        fcl_out = self.fcl(encoded)
        # Final prediction
        return self.head(fcl_out.unsqueeze(-1)).squeeze(-1)


__all__ = ["HybridRegression", "HybridDataset", "generate_hybrid_data"]
