"""Quantum regression model built with Pennylane.

The implementation follows the original seed but introduces a
parameterised variational circuit with entanglement and multiple
measurement observables.  The circuit is wrapped in a QNode and
combined with a classical head, mirroring the API of the classical
model for seamless comparison.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pennylane as qml

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate states |ψ(θ,φ)〉 = cosθ|0…0〉 + e^{iφ} sinθ|1…1〉.

    The labels are sin(2θ)cosφ, matching the seed but implemented
    with Pennylane's numpy backend for compatibility.
    """
    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2**num_wires), dtype=complex)
    for i in range(samples):
        state_vec = np.zeros(2**num_wires, dtype=complex)
        state_vec[0] = np.cos(thetas[i])
        state_vec[-1] = np.exp(1j * phis[i]) * np.sin(thetas[i])
        states[i] = state_vec
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset that yields quantum states and targets.

    The states are stored as complex tensors and can be fed into a
    Pennylane QNode via a ``QuantumDevice``.
    """

    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class QuantumRegressionModel(nn.Module):
    """Quantum variational regression model with a classical head."""

    def __init__(self, num_wires: int, n_layers: int = 3):
        super().__init__()
        self.num_wires = num_wires
        self.n_layers = n_layers

        # Define a variational circuit
        dev = qml.device("default.qubit", wires=num_wires, shots=None)

        @qml.qnode(dev, interface="torch")
        def circuit(state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
            # Encode the classical state via a simple angle encoding
            for i in range(num_wires):
                qml.RY(state[i], wires=i)
            # Entangling layers
            for _ in range(n_layers):
                for i in range(num_wires - 1):
                    qml.CNOT(wires=[i, i + 1])
                for i in range(num_wires):
                    qml.RY(params[i], wires=i)
            # Measure expectation values of PauliZ on all wires
            return [qml.expval(qml.PauliZ(i)) for i in range(num_wires)]

        self.circuit = circuit
        # Trainable parameters for the variational layers
        self.params = nn.Parameter(torch.randn(num_wires))
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        # state_batch shape: (batch, num_wires)
        features = torch.stack([self.circuit(state, self.params) for state in state_batch], dim=0)
        return self.head(features).squeeze(-1)

__all__ = ["QuantumRegressionModel", "RegressionDataset", "generate_superposition_data"]
