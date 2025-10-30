"""Quantum regression model using Pennylane variational circuit."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from pennylane import qnn


def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample states of the form cos(theta)|0..0> + e^{i phi} sin(theta)|1..1>.
    The labels are derived from the angles to provide a non‑linear regression task.
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
    """Dataset wrapping the quantum superposition states."""
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
    """Quantum regression model built with Pennylane's TorchLayer."""
    def __init__(self, num_wires: int):
        super().__init__()
        self.num_wires = num_wires
        self.dev = qml.device("default.qubit", wires=num_wires, shots=None)
        self.qnn = qnn.TorchLayer(self._circuit, output_dim=num_wires)
        self.head = nn.Linear(num_wires, 1)

    def _circuit(self, x: torch.Tensor, params: torch.Tensor) -> list[torch.Tensor]:
        """
        Variational circuit with amplitude embedding, local rotations and a
        linear entangling layer.  It returns expectation values of Pauli‑Z.
        """
        qml.AmplitudeEmbedding(features=x, wires=range(self.num_wires), normalize=True)
        for i in range(self.num_wires):
            qml.RX(params[0, i], wires=i)
            qml.RY(params[1, i], wires=i)
        for i in range(self.num_wires - 1):
            qml.CNOT(wires=[i, i + 1])
        return [qml.expval(qml.PauliZ(w)) for w in range(self.num_wires)]

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Encode the complex state amplitudes into a real vector, run the
        variational circuit in parallel over the batch and linearly regress
        the expectation values.
        """
        features = torch.abs(state_batch)
        output = self.qnn(features)
        return self.head(output).squeeze(-1)


__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
