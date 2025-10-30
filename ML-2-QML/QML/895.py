"""Quantum regression model using Pennylane's hybrid circuit."""

import pennylane as qml
import pennylane.numpy as pnp
import torch
import torch.nn as nn
import numpy as np


def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic quantum states of the form
    cos(theta)|0...0⟩ + e^{i phi} sin(theta)|1...1⟩.

    Parameters
    ----------
    num_wires : int
        Number of qubits in each state.
    samples : int
        Number of samples to generate.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        States of shape (samples, 2**num_wires) and labels of shape (samples,).
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
    PyTorch dataset wrapping the synthetic quantum regression data.
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
    Hybrid quantum-classical regression model built with Pennylane.
    """

    def __init__(self, num_wires: int):
        super().__init__()
        self.num_wires = num_wires
        self.dev = qml.device("default.qubit", wires=num_wires, shots=None)
        # Trainable parameters for single-qubit rotations
        self.params = nn.Parameter(torch.randn(num_wires, 3))
        # Trainable parameters for entangling rotations
        self.ent_params = nn.Parameter(torch.randn(num_wires, 3))
        self.classical_head = nn.Linear(num_wires, 1)
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")

    def _circuit(self, state: torch.Tensor, params: torch.Tensor, ent_params: torch.Tensor):
        qml.StatePrep(state, wires=range(self.num_wires))
        for i in range(self.num_wires):
            qml.RX(params[i, 0], wires=i)
            qml.RY(params[i, 1], wires=i)
            qml.RZ(params[i, 2], wires=i)
        # Simple entangling layer
        for i in range(self.num_wires - 1):
            qml.CNOT(wires=[i, i + 1])
        return [qml.expval(qml.PauliZ(i)) for i in range(self.num_wires)]

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        # state_batch shape (batch, 2**num_wires)
        batch_size = state_batch.shape[0]
        state_batch = state_batch.to(torch.complex64)
        # Apply the circuit in a batched fashion
        qout = torch.vmap(lambda s: self.qnode(s, self.params, self.ent_params))(state_batch)
        return self.classical_head(qout).squeeze(-1)


__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
