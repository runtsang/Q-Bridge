"""Quantum regression dataset and model derived from ``new_run_regression.py`` with a PennyLane variational circuit."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
import pennylane.numpy as npq  # PennyLane's autograd-compatible NumPy


def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample states of the form cos(theta)|0..0⟩ + e^{i phi} sin(theta)|1..1⟩."""
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
    return states, labels


class RegressionDataset(torch.utils.data.Dataset):
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
    Hybrid model that encodes input states into a quantum device, runs a parameter‑shared variational circuit,
    extracts expectation values, and feeds them into a classical linear head.
    """
    def __init__(self, num_wires: int, num_layers: int = 3, device_name: str = "default.qubit"):
        super().__init__()
        self.num_wires = num_wires
        self.dev = qml.device(device_name, wires=num_wires, shots=None)

        # Parameters for the variational circuit
        self.params = nn.Parameter(torch.randn(num_layers, num_wires) * 0.1)

        # Classical head
        self.head = nn.Linear(num_wires, 1)

        # Compile the QNode once
        @qml.qnode(self.dev, interface="torch", batch_mode=True)
        def _qnode(state_batch: torch.Tensor):
            # Encode each batch element as amplitude state
            qml.StatePreparation(state=state_batch, wires=range(num_wires))
            for layer in range(num_layers):
                for wire in range(num_wires):
                    qml.RY(self.params[layer, wire], wires=wire)
                # Entangling layer
                for wire in range(num_wires - 1):
                    qml.CNOT(wires=[wire, wire + 1])
            return [qml.expval(qml.PauliZ(w)) for w in range(num_wires)]

        self._qnode = _qnode

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: run the quantum circuit on the batch of complex states
        and apply the linear head.
        """
        q_features = self._qnode(state_batch)
        return self.head(q_features).squeeze(-1)


__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
