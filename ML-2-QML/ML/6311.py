"""Hybrid fully‑connected layer with classical implementation.

This module extends the original FCL example by adding a quantum‑inspired
feature map and dataset utilities from the QuantumRegression seed.
The class `HybridFCL` behaves like a normal PyTorch module but also
provides a `run` method that mimics the quantum circuit expectation
value for a given set of parameters.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a toy regression dataset where the target is a nonlinear
    function of the sum of input features.  This mirrors the
    `generate_superposition_data` from the quantum seed.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """
    Dataset that returns a dictionary with keys ``states`` and ``target``.
    ``states`` are the raw input features; ``target`` is the regression label.
    """
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class HybridFCL(nn.Module):
    """
    Classical hybrid fully‑connected layer.

    The module first applies a linear transformation to the input,
    then passes the result through a pair of quantum‑style nonlinearities
    (sigmoid and tanh).  The concatenated features are returned.
    """
    def __init__(self, n_features: int = 1, n_qubits: int = 1):
        super().__init__()
        self.linear = nn.Linear(n_features, n_qubits)
        self.feature_map = nn.ModuleList([nn.Sigmoid(), nn.Tanh()])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.linear(x)
        features = [layer(z) for layer in self.feature_map]
        return torch.cat(features, dim=-1)

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """
        Mimic a quantum circuit expectation value by applying the linear layer
        to the parameters and computing the mean of a tanh nonlinearity.
        """
        thetas = torch.as_tensor(thetas, dtype=torch.float32).view(-1, 1)
        z = self.linear(thetas)
        expectation = torch.tanh(z).mean(dim=0)
        return expectation.detach().cpu().numpy()


def FCL() -> HybridFCL:
    """
    Factory function that returns an instance of the hybrid layer.
    """
    return HybridFCL()
