"""Quantum regression model using Pennylane with a variational circuit and a classical head."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as pnp

__all__ = ["QuantumRegressionModel", "RegressionDataset", "generate_superposition_data"]

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate superposition states |ψ(θ, φ)⟩ = cosθ|0…0⟩ + e^{iφ} sinθ|1…1⟩ and labels."""
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

class RegressionDataset(torch.utils.data.Dataset):
    """Dataset wrapper that returns quantum states as complex tensors and real targets."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int) -> dict:
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QuantumRegressionModel(nn.Module):
    """Hybrid quantum‑classical regression model."""
    def __init__(self, num_wires: int, hidden_dim: int = 32):
        super().__init__()
        self.num_wires = num_wires
        self.dev = qml.device("default.qubit", wires=num_wires)
        # Variational parameters
        self.params = nn.Parameter(torch.randn(num_wires, 3))
        # Classical readout head
        self.head = nn.Linear(num_wires, 1)

    def circuit(self, state, params):
        """Angle‑encoded input followed by a 1‑layer variational circuit."""
        qml.QubitStateVector(state, wires=range(self.num_wires))
        for i in range(self.num_wires):
            qml.RX(params[i, 0], wires=i)
            qml.RZ(params[i, 1], wires=i)
            qml.RX(params[i, 2], wires=i)
        return [qml.expval(qml.PauliZ(i)) for i in range(self.num_wires)]

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def qnode(state):
            return self.circuit(state, self.params)
        features = qnode(state_batch)
        return self.head(features).squeeze(-1)
