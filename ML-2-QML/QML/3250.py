"""Hybrid sampler and regression model implemented with Pennylane.

The quantum version reproduces the classical sampler using a parameterised
circuit and replaces the regression head with a variational circuit
followed by a classical linear layer.  The API remains identical to the
PyTorch implementation for seamless interchangeability.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate complex superposition states and labels."""
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset wrapping the complex superposition states."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class HybridSamplerRegressor:
    """Quantum sampler followed by a variational regression circuit."""
    def __init__(self, num_wires: int = 2):
        self.num_wires = num_wires

        # Devices
        self.sampler_dev = qml.device("default.qubit", wires=num_wires)
        self.regress_dev = qml.device("default.qubit", wires=num_wires)

        # Parameters
        self.sampler_weights = np.random.randn(num_wires, 2)  # 2 Ry gates per wire
        self.encode_params = np.random.randn(num_wires * 2)   # 2 rotations per wire
        self.random_layer_params = np.random.randn(num_wires * 3)  # 3 rotations per wire

        # Sampler circuit
        @qml.qnode(self.sampler_dev, interface="torch")
        def sampler_qnode(x: torch.Tensor, w: np.ndarray):
            for i in range(self.num_wires):
                qml.RY(x[i], wires=i)
            qml.CNOT(wires=[0, 1])
            for i in range(self.num_wires):
                qml.RY(w[i], wires=i)
            return qml.state()

        # Regression circuit
        @qml.qnode(self.regress_dev, interface="torch")
        def regress_qnode(state: torch.Tensor):
            qml.QubitStateVector(state, wires=range(self.num_wires))
            for i in range(self.num_wires):
                qml.RX(self.encode_params[i], wires=i)
                qml.RY(self.encode_params[self.num_wires + i], wires=i)
            # Simple random layer
            for i in range(self.num_wires):
                qml.RZ(self.random_layer_params[i], wires=i)
                qml.RX(self.random_layer_params[self.num_wires + i], wires=i)
                qml.RY(self.random_layer_params[2 * self.num_wires + i], wires=i)
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_wires)]

        self.sampler_qnode = sampler_qnode
        self.regress_qnode = regress_qnode

        # Classical head
        self.head = nn.Linear(num_wires, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass: sampler → regression → linear head."""
        state = self.sampler_qnode(z, self.sampler_weights)
        features = self.regress_qnode(state)
        return self.head(torch.tensor(features)).squeeze(-1)

__all__ = ["HybridSamplerRegressor", "RegressionDataset", "generate_superposition_data"]
