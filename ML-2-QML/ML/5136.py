"""Hybrid fully‑connected layer with optional quantum expectation head.

This module unites the classical fully‑connected architecture from the
original FCL example with a flexible interface for a quantum circuit.
The network can be used for regression or binary classification.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from typing import Iterable, List, Optional


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate a toy regression dataset where the target is a nonlinear function
    of a linear combination of the input features.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset wrapper that returns feature vectors and target scalars."""

    def __init__(self, samples: int, num_features: int) -> None:
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class HybridFullyConnectedLayer(nn.Module):
    """
    Classical fully‑connected network that optionally forwards its activations
    through a quantum circuit via the ``quantum_circuit`` callable.
    """

    def __init__(
        self,
        num_features: int,
        hidden_sizes: List[int] = [32, 16],
        output_size: int = 1,
        quantum_circuit: Optional[callable] = None,
    ) -> None:
        super().__init__()
        layers = []
        prev = num_features
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_size))
        self.net = nn.Sequential(*layers)
        self.quantum_circuit = quantum_circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out = self.net(x)
        if self.quantum_circuit is not None:
            # The quantum circuit expects a 1‑D array of angles.
            angles = out.squeeze().detach().cpu().numpy()
            out = torch.tensor(self.quantum_circuit(angles), dtype=torch.float32, device=x.device)
        return out.squeeze(-1)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Mimic the original FCL interface: compute a scalar expectation
        from a sequence of angles using a simple linear layer.
        """
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.net(values)).mean(dim=0)
        return expectation.detach().numpy()


__all__ = [
    "HybridFullyConnectedLayer",
    "RegressionDataset",
    "generate_superposition_data",
]
