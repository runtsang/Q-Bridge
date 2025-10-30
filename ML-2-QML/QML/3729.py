"""Quantum hybrid kernel estimator with optional classical feature mapping.

Combines a TorchQuantum ansatz with a small neural network inspired by EstimatorQNN.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from typing import Sequence

class FeatureNet(nn.Module):
    """Optional classical feature extractor."""
    def __init__(self, input_dim: int, hidden_dim: int = 8, output_dim: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class QuantumKernelAnsatz(tq.QuantumModule):
    """Programmable quantum circuit encoding two vectors."""
    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for i in range(self.n_wires):
            param = x[:, i] if x.shape[1] > i else None
            tq.ry(q_device, wires=[i], params=param)
        for i in reversed(range(self.n_wires)):
            param = -y[:, i] if y.shape[1] > i else None
            tq.ry(q_device, wires=[i], params=param)

class HybridKernelEstimator(tq.QuantumModule):
    """Hybrid quantum kernel with optional classical feature net."""
    def __init__(self, input_dim: int, n_wires: int = 4) -> None:
        super().__init__()
        self.feature_net = FeatureNet(input_dim)
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = QuantumKernelAnsatz(n_wires=self.n_wires)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.feature_net(x)
        y = self.feature_net(y)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        mat = torch.stack([self.forward(x, y) for x in a for y in b])
        return mat.view(len(a), len(b)).numpy()

__all__ = ["HybridKernelEstimator", "FeatureNet", "QuantumKernelAnsatz"]
