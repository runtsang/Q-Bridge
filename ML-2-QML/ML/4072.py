"""Hybrid classical regression model combining encoder, quantum-inspired fully connected layer,
and a classical sampler network.

The module follows the structure of the original QuantumRegression seed but augments it with
three additional components:

* `FCL` – a lightweight fully connected layer that mimics a quantum expectation value
  by applying a tanh non‑linearity to a single weight.
* `SamplerQNN` – a small softmax network that emulates a Qiskit sampler; its output is
  treated as a probabilistic feature vector.
* `HybridRegression` – a PyTorch model that concatenates the outputs of the encoder,
  the quantum‑inspired layer, and the sampler before feeding them to a final linear head.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

__all__ = ["generate_superposition_data", "RegressionDataset", "HybridRegression"]


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic regression dataset where the target is a noisy sinusoid
    of the sum of the input features.  The function is identical to the seed
    implementation but the type hints are tightened for clarity.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Simple PyTorch dataset wrapping the synthetic data."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


def FCL() -> nn.Module:
    """Return a tiny fully‑connected layer that emulates a quantum expectation value."""
    class FullyConnectedLayer(nn.Module):
        def __init__(self, n_features: int = 1) -> None:
            super().__init__()
            self.linear = nn.Linear(n_features, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.tanh(self.linear(x))
    return FullyConnectedLayer()


def SamplerQNN() -> nn.Module:
    """Return a very small neural network that mimics the behaviour of a quantum sampler."""
    class SamplerModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 4),
                nn.Tanh(),
                nn.Linear(4, 2),
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return F.softmax(self.net(inputs), dim=-1)
    return SamplerModule()


class HybridRegression(nn.Module):
    """Classical regression model that fuses three distinct feature generators."""
    def __init__(self, num_features: int):
        super().__init__()
        self.encoder = nn.Linear(num_features, num_features * 2)
        self.fcl = FCL()
        self.sampler = SamplerQNN()
        self.head = nn.Linear(num_features * 2 + 1 + 2, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        enc = self.encoder(state_batch)
        fcl_out = self.fcl(state_batch[:, 0:1])
        sampler_out = self.sampler(state_batch[:, :2])
        features = torch.cat([enc, fcl_out, sampler_out], dim=-1)
        return self.head(features).squeeze(-1)
