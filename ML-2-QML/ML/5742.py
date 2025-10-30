"""Hybrid fully connected layer combining classical neural network and quantum‑inspired expectation."""

from __future__ import annotations

import torch
from torch import nn
from typing import Iterable
import numpy as np


class HybridFullyConnectedLayer(nn.Module):
    """
    A hybrid layer that processes inputs through a small classical network
    and then feeds the result into a quantum‑inspired expectation layer.

    The quantum part is emulated with a parameterised tanh function, mimicking
    the expectation value of a Pauli‑Y measurement on a parameterised qubit.

    The design borrows from:
    - FCL.py: the tanh‑based expectation layer
    - EstimatorQNN.py: the multi‑layer classical feed‑forward structure
    """

    def __init__(self, n_features: int = 1, hidden_sizes: Iterable[int] = (8, 4)) -> None:
        super().__init__()
        layers = []
        input_dim = n_features
        for size in hidden_sizes:
            layers.append(nn.Linear(input_dim, size))
            layers.append(nn.Tanh())
            input_dim = size
        layers.append(nn.Linear(input_dim, 1))
        self.classical_net = nn.Sequential(*layers)

        # Parameter mimicking the quantum weight
        self.quantum_weight = nn.Parameter(torch.randn(1, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        classical_out = self.classical_net(x)
        # Quantum‑inspired expectation: tanh(weight * classical_out)
        quantum_out = torch.tanh(self.quantum_weight * classical_out)
        return quantum_out


__all__ = ["HybridFullyConnectedLayer"]
