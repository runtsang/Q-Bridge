"""Quantum regression model implemented with PennyLane and PyTorch."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate states of the form cos(theta)|0…0> + e^{i phi} sin(theta)|1…1>.
    The target is a non‑linear function of theta and phi.
    """
    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        omega0 = np.zeros(2 ** num_wires, dtype=complex)
        omega0[0] = 1.0
        omega1 = np.zeros(2 ** num_wires, dtype=complex)
        omega1[-1] = 1.0
        states[i] = np.cos(thetas[i]) * omega0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels.astype(np.float32)

class RegressionDataset(torch.utils.data.Dataset):
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class QModel(nn.Module):
    """
    Hybrid quantum‑classical regression model.

    The quantum circuit consists of alternating rotation and entangling layers.
    Expectation values of Pauli‑Z are used as features for a linear read‑out.
    """
    def __init__(self, num_wires: int, num_layers: int = 2):
        super().__init__()
        self.num_wires = num_wires
        self.num_layers = num_layers

        # PennyLane device with batch support
        self.dev = qml.device("default.qubit", wires=num_wires, shots=None)

        @qml.qnode(self.dev, interface="torch")
        def circuit(state: torch.Tensor, params: torch.Tensor):
            # Encode the classical state into rotations
            for w in range(num_wires):
                qml.RY(state[w], wires=w)
            # Parameterized layers
            for layer in range(num_layers):
                for w in range(num_wires):
                    qml.RY(params[layer, w, 0], wires=w)
                    qml.RZ(params[layer, w, 1], wires=w)
                # Entangling layer
                for w in range(num_wires - 1):
                    qml.CNOT(wires=[w, w + 1])
                qml.CNOT(wires=[num_wires - 1, 0])
            return [qml.expval(qml.PauliZ(w)) for w in range(num_wires)]

        self.circuit = circuit
        self.params = nn.Parameter(torch.rand(num_layers, num_wires, 2))

        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        state_batch : torch.Tensor
            Tensor of shape (batch, 2**num_wires) holding the amplitude
            representation of each state.  The first ``num_wires`` real
            components are used as rotation angles for the encoder.
        Returns
        -------
        torch.Tensor
            Predicted regression values of shape (batch,).
        """
        batch_size = state_batch.shape[0]
        angles = torch.real(state_batch)[:, :self.num_wires]
        features = torch.stack(
            [self.circuit(angles[i], self.params) for i in range(batch_size)],
            dim=0,
        )
        return self.head(features).squeeze(-1)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
