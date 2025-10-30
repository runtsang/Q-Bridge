\
"""Quantum regression module using Pennylane variational circuits."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as pnp

def generate_superposition_data(
    num_wires: int,
    samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample states of the form cos(theta)|0...0> + e^{i phi} sin(theta)|1...1>.
    Returns the corresponding labels.
    """
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
    """
    Dataset wrapper that returns a dictionary of tensors.
    """
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
    Hybrid quantum‑classical model with a variational circuit and a linear head.
    """
    def __init__(self, num_wires: int, depth: int = 2):
        super().__init__()
        self.num_wires = num_wires
        self.depth = depth

        # Quantum device
        self.dev = qml.device("default.qubit", wires=num_wires)

        # Trainable parameters for the variational circuit
        self.params = nn.Parameter(torch.randn(depth, num_wires, 3) * 0.1)

        # Classical head
        self.head = nn.Linear(num_wires, 1)

    def quantum_circuit(self, x, params):
        """
        Variational circuit with tunable depth.
        x: array of shape (num_wires,)
        params: array of shape (depth, num_wires, 3)
        """
        for d in range(self.depth):
            for w in range(self.num_wires):
                qml.RX(params[d, w, 0], wires=w)
                qml.RY(params[d, w, 1], wires=w)
                qml.RZ(params[d, w, 2], wires=w)
            # Entangling layer
            for w in range(self.num_wires - 1):
                qml.CNOT(wires=[w, w + 1])
        return [qml.expval(qml.PauliZ(w)) for w in range(self.num_wires)]

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass over a batch of complex state vectors.
        """
        # Convert to numpy array for Pennylane
        states_np = state_batch.cpu().numpy()
        batch_size = states_np.shape[0]
        features = np.zeros((batch_size, self.num_wires), dtype=np.float32)

        # Encode each state into angles for the variational circuit
        for i in range(batch_size):
            amps = states_np[i]
            phases = np.angle(amps)
            phases = (phases + np.pi) % (2 * np.pi)
            features[i] = phases[:self.num_wires]

        quantum_out = []
        for i in range(batch_size):
            qnode = qml.QNode(
                lambda x: self.quantum_circuit(x, self.params),
                self.dev,
                interface="torch",
            )
            qnode_out = qnode(torch.tensor(features[i], dtype=torch.float32))
            quantum_out.append(qnode_out)
        quantum_out = torch.stack(quantum_out)

        # Classical head
        return self.head(quantum_out).squeeze(-1)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
