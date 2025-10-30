"""Quantum regression using a Pennylane hybrid circuit with CZ entanglement."""
from __future__ import annotations

import numpy as np
import pennylane as qml
import torch
from torch import nn
from torch.utils.data import Dataset


def generate_superposition_data(num_wires: int, samples: int,
                                noise_std: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate data for a quantum regression task.
    States are of the form |ψ⟩ = cosθ|0…0⟩ + e^{iφ} sinθ|1…1⟩.
    The target is a smooth function of θ and φ with optional noise.
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
    if noise_std > 0.0:
        labels += np.random.normal(scale=noise_std, size=labels.shape)
    return states, labels.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset wrapper for quantum regression data."""

    def __init__(self, samples: int, num_wires: int,
                 noise_std: float = 0.0):
        self.states, self.labels = generate_superposition_data(
            num_wires, samples, noise_std
        )

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class RegressionModel(nn.Module):
    """
    Hybrid quantum‑classical regression model.

    The circuit encodes the amplitude‑encoded state, applies a parameterised
    entangling layer (CZ), and measures expectation values of Pauli‑Z on each
    qubit.  A classical linear head maps the resulting feature vector to the
    scalar output.
    """

    def __init__(self, num_wires: int, device: str = "default.qubit"):
        super().__init__()
        self.n_wires = num_wires
        self.dev = qml.device(device, wires=num_wires, shots=None)

        # Trainable circuit parameters
        self.params = nn.Parameter(torch.randn(num_wires, 3))

        # Classical linear head
        self.head = nn.Linear(num_wires, 1)

        # Define the quantum node
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(x: torch.Tensor):
            # Encode amplitude‑encoded state via rotation on a dummy qubit
            for i in range(num_wires):
                qml.RY(x[i], wires=i)
            # Entangling layer
            for i in range(num_wires - 1):
                qml.CZ(wires=[i, i + 1])
            # Parameterised rotation layer
            for i in range(num_wires):
                qml.RX(self.params[i, 0], wires=i)
                qml.RY(self.params[i, 1], wires=i)
                qml.RZ(self.params[i, 2], wires=i)
            # Measure expectation values of Pauli‑Z
            return [qml.expval(qml.PauliZ(i)) for i in range(num_wires)]

        self.circuit = circuit

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        # Convert amplitude‑encoded complex state to real rotation angles
        # Here we simply use the real part of the state as a proxy for angle
        angles = torch.real(state_batch)
        # Normalise angles to [-π, π]
        angles = (angles - angles.min()) / (angles.max() - angles.min()) * 2 * np.pi - np.pi
        q_out = self.circuit(angles)
        return self.head(q_out).squeeze(-1)


__all__ = ["RegressionModel", "RegressionDataset", "generate_superposition_data"]
