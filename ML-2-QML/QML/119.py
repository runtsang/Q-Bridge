"""Quantum regression dataset and hybrid variational circuit."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml


def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Create states |ψ(θ,ϕ)⟩ = cosθ|0…0⟩ + e^{iϕ}sinθ|1…1⟩
    with a more challenging target function.
    """
    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    dim = 2 ** num_wires
    states = np.zeros((samples, dim), dtype=complex)
    for i in range(samples):
        omega0 = np.zeros(dim, dtype=complex)
        omega0[0] = 1.0
        omega1 = np.zeros(dim, dtype=complex)
        omega1[-1] = 1.0
        states[i] = np.cos(thetas[i]) * omega0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega1
    labels = np.sin(3 * thetas) * np.cos(2 * phis) + 0.1 * np.random.randn(samples)
    return states, labels.astype(np.float32)


class RegressionDataset(torch.utils.data.Dataset):
    """Dataset returning complex state vectors and targets."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int):
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class SharedRegressionModel(nn.Module):
    """
    Hybrid quantum‑classical regression model.
    Uses a Pennylane QNode with trainable RX/RY layers and CNOT entanglement.
    The circuit outputs Pauli‑Z expectation values for each qubit,
    which are passed through a linear read‑out head.
    """
    def __init__(self, num_wires: int, num_layers: int = 3, device: str = "default.qubit"):
        super().__init__()
        self.num_wires = num_wires
        self.num_layers = num_layers
        dev = qml.device(device, wires=num_wires, shots=1)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            # Encode classical data as rotations
            for i in range(num_wires):
                qml.RX(inputs[i], wires=i)
                qml.RY(inputs[i], wires=i)
            # Variational layers
            for l in range(num_layers):
                for i in range(num_wires):
                    qml.RX(weights[l, i, 0], wires=i)
                    qml.RY(weights[l, i, 1], wires=i)
                for i in range(num_wires - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[num_wires - 1, 0])  # wrap‑around entanglement
            return [qml.expval(qml.PauliZ(i)) for i in range(num_wires)]

        self.circuit = circuit
        self.weight = nn.Parameter(torch.randn(num_layers, num_wires, 2))
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        state_batch: Tensor of shape (batch, 2**num_wires) with complex dtype.
        Convert to rotation angles and evaluate the circuit sample‑wise.
        """
        outputs = []
        for i in range(state_batch.shape[0]):
            angles = torch.atan2(state_batch[i].imag, state_batch[i].real)
            inputs = angles[:self.num_wires]
            q_out = self.circuit(inputs, self.weight)
            outputs.append(self.head(torch.tensor(q_out, device=state_batch.device)))
        return torch.cat(outputs, dim=0).squeeze(-1)


__all__ = ["SharedRegressionModel", "RegressionDataset", "generate_superposition_data"]
