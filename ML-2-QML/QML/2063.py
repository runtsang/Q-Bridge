"""Quantum regression model using PennyLane and a variational circuit."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as pnp
from torch.utils.data import Dataset


def generate_superposition_data(
    num_wires: int, samples: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create superposition states |ψ(θ,φ)⟩ = cosθ|0…0⟩ + e^{iφ} sinθ|1…1⟩
    and a target y = sin(2θ) cosφ.  The states are returned as complex
    float64 arrays for compatibility with PennyLane.
    """
    # Basis vectors for |0…0⟩ and |1…1⟩
    zero_state = np.zeros(2 ** num_wires, dtype=complex)
    zero_state[0] = 1.0
    one_state = np.zeros(2 ** num_wires, dtype=complex)
    one_state[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)

    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * zero_state + np.exp(1j * phis[i]) * np.sin(thetas[i]) * one_state

    labels = np.sin(2 * thetas) * np.cos(phis)
    return states.astype(np.complex64), labels.astype(np.float32)


class RegressionDataset(Dataset):
    """Torch dataset yielding quantum state tensors and scalar targets."""

    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class QModel(nn.Module):
    """
    Hybrid quantum‑classical regression model.  The variational circuit
    is implemented with PennyLane's QNode, and the output of the circuit
    is fed into a small classical head.
    """

    def __init__(self, num_wires: int, hidden: int = 8):
        super().__init__()
        self.num_wires = num_wires

        # PennyLane device and QNode
        self.dev = qml.device("default.qubit", wires=num_wires)
        self.qnode = qml.qnode(self.dev, interface="torch", diff_method="backprop")

        # Trainable parameters of the variational circuit
        self.params = nn.Parameter(torch.randn(num_wires, dtype=torch.float32))

        # Classical head
        self.head = nn.Linear(num_wires, hidden)
        self.out = nn.Linear(hidden, 1)
        self.relu = nn.ReLU()

    def circuit(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Variational circuit:
          - Encode each feature as an RX rotation.
          - Apply a layer of RX/RY rotations with trainable angles.
          - Entangle adjacent qubits with CNOTs.
          - Measure Pauli‑Z expectation values on all wires.
        """
        for i in range(self.num_wires):
            qml.RX(x[i], wires=i)
        for i in range(self.num_wires):
            qml.RY(params[i], wires=i)
        for i in range(self.num_wires - 1):
            qml.CNOT(wires=[i, i + 1])
        return [qml.expval(qml.PauliZ(i)) for i in range(self.num_wires)]

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass over a batch of quantum states.
        The QNode automatically vectorises over the batch dimension.
        """
        # state_batch shape: (batch, num_wires)
        q_values = self.qnode(state_batch, self.params)  # shape: (batch, num_wires)
        features = self.relu(self.head(q_values))
        return self.out(features).squeeze(-1)


__all__ = ["RegressionDataset", "QModel", "generate_superposition_data"]
