"""Hybrid classical regression model with optional quantum feature extraction.

This module extends the original QuantumRegression example by adding:
* A flexible dataset generator that normalises features.
* A fully connected layer wrapper (FCL) that can be used as a drop‑in.
* A QuantumHybridRegression class that can operate purely classically or
  delegate feature extraction to a user supplied quantum module.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Optional

# --------------------------------------------------------------------------- #
# Dataset utilities
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data for regression.

    The target is a smooth function of the sum of features, mimicking a
    superposition of basis states.  Features are normalised to zero mean
    and unit variance to aid optimisation.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    # Normalise
    x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-6)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """PyTorch dataset that yields feature tensors and target scalars."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# --------------------------------------------------------------------------- #
# Fully connected layer wrapper
# --------------------------------------------------------------------------- #
def FCL(n_features: int = 1) -> nn.Module:
    """Return a simple fully connected layer with a run method.

    This mirrors the quantum implementation in the seed while remaining
    purely classical.  It can be swapped in as a drop‑in replacement for
    quantum layers in the hybrid model.
    """
    class FullyConnectedLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(n_features, 1)

        def run(self, thetas: np.ndarray) -> np.ndarray:
            values = torch.as_tensor(thetas, dtype=torch.float32).view(-1, 1)
            expectation = torch.tanh(self.linear(values)).mean(dim=0)
            return expectation.detach().numpy()

    return FullyConnectedLayer()

# --------------------------------------------------------------------------- #
# Hybrid regression model
# --------------------------------------------------------------------------- #
class QuantumHybridRegression(nn.Module):
    """Hybrid model that optionally delegates feature extraction to a quantum module.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input feature vector.
    use_quantum : bool, default False
        If True, the model expects a quantum module that implements a
        ``forward`` method returning a tensor of shape (batch, num_wires).
    quantum_module : nn.Module, optional
        The quantum module to use when ``use_quantum`` is True.  It must
        accept a tensor of shape (batch, num_features) and return a tensor
        of shape (batch, num_wires).
    dropout : float, default 0.0
        Dropout probability applied after the encoder (classical or quantum).
    """
    def __init__(
        self,
        num_features: int,
        use_quantum: bool = False,
        quantum_module: Optional[nn.Module] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.use_quantum = use_quantum
        if self.use_quantum:
            if quantum_module is None:
                raise ValueError("quantum_module must be provided when use_quantum=True")
            self.encoder = quantum_module
            # Infer feature dimension from a dummy forward pass
            with torch.no_grad():
                dummy = torch.zeros(1, num_features, device=next(quantum_module.parameters()).device)
                out = self.encoder(dummy)
                self.feature_dim = out.shape[-1]
        else:
            self.encoder = nn.Sequential(
                nn.Linear(num_features, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(32),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.feature_dim = 16
        self.head = nn.Linear(self.feature_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        If ``use_quantum`` is True the input ``x`` is forwarded to the
        user supplied quantum module.  The returned tensor is then
        processed by the classical head.
        """
        if self.use_quantum:
            features = self.encoder(x)
        else:
            features = self.encoder(x)
        return self.head(features).squeeze(-1)

__all__ = ["QuantumHybridRegression", "RegressionDataset", "generate_superposition_data", "FCL"]
