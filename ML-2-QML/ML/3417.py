"""Hybrid classical regression module.

This module implements a deep fully‑connected neural network that can optionally
be coupled to a quantum feature extractor.  It also contains utilities for
generating the superposition dataset used in the quantum reference.

The classical backbone mirrors the original `FCL.py` example but with
additional hidden layers, batch‑normalisation and a flexible interface for
a quantum module.  The design demonstrates how a purely classical network can
be prepared to receive quantum‑derived features in a hybrid setting.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from typing import Optional

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic dataset of superposition states and a target.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input feature vectors.
    samples : int
        Number of samples to generate.

    Returns
    -------
    X : np.ndarray
        Uniformly sampled feature matrix of shape (samples, num_features).
    y : np.ndarray
        Target values computed as ``sin(sum(x)) + 0.1*cos(2*sum(x))``.
    """
    X = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = X.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return X, y.astype(np.float32)

class RegressionDataset(Dataset):
    """PyTorch dataset that returns the amplitude‑encoded data and target."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class HybridFCLRegression(nn.Module):
    """
    Classical fully‑connected regression model with optional quantum feature input.

    The network is a standard multi‑layer perceptron.  When a quantum module is
    supplied, the quantum output is concatenated to the classical representation
    before the final regression head.

    Parameters
    ----------
    num_features : int
        Dimensionality of the classical input.
    hidden_dim : int, optional
        Size of the hidden layer preceding the final head.
    quantum_module : nn.Module, optional
        A callable accepting a batch of quantum features and returning a tensor.
    """
    def __init__(self, num_features: int, hidden_dim: int = 32,
                 quantum_module: Optional[nn.Module] = None):
        super().__init__()
        self.encoder = nn.Linear(num_features, hidden_dim)
        self.activation = nn.ReLU()
        self.hidden = nn.Linear(hidden_dim, 16)
        self.output = nn.Linear(16, 1)
        self.quantum_module = quantum_module

    def forward(self, x: torch.Tensor, quantum_input: torch.Tensor | None = None) -> torch.Tensor:
        x = self.activation(self.encoder(x))
        if self.quantum_module is not None and quantum_input is not None:
            q_feat = self.quantum_module(quantum_input)
            x = torch.cat([x, q_feat], dim=-1)
        x = self.activation(self.hidden(x))
        return self.output(x).squeeze(-1)

__all__ = ["HybridFCLRegression", "RegressionDataset", "generate_superposition_data"]
