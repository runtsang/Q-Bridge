"""Quantum regression model using PennyLane."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate amplitude‑encoded superposition states."""
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
    """Dataset providing amplitude‑encoded states and labels."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QLayer(nn.Module):
    """Variational quantum layer with entanglement."""
    def __init__(self, num_wires: int, num_layers: int = 3):
        super().__init__()
        self.num_wires = num_wires
        self.num_layers = num_layers
        self.params = nn.Parameter(torch.randn(num_layers, num_wires, 2))

    def forward(self, state_vector: torch.Tensor) -> torch.Tensor:
        dev = qml.device("default.qubit", wires=self.num_wires)
        @qml.qnode(dev, interface="torch")
        def circuit(vec, params):
            qml.QubitStateVector(vec, wires=range(self.num_wires))
            for l in range(self.num_layers):
                for w in range(self.num_wires):
                    qml.RX(params[l, w, 0], wires=w)
                    qml.RY(params[l, w, 1], wires=w)
                for w in range(self.num_wires - 1):
                    qml.CNOT(wires=[w, w + 1])
            return [qml.expval(qml.PauliZ(w)) for w in range(self.num_wires)]

        batch_size = state_vector.shape[0]
        outputs = torch.stack([circuit(state_vector[i], self.params) for i in range(batch_size)])
        return outputs

class QModel(nn.Module):
    """Quantum regression model."""
    def __init__(self, num_wires: int, hidden_dim: int = 8):
        super().__init__()
        self.num_wires = num_wires
        self.q_layer = QLayer(num_wires)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        features = self.q_layer(state_batch)
        return self.head(features).squeeze(-1)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
