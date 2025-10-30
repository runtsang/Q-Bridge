"""Quantum regression using PennyLane."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as pnp

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate states of the form cos(theta)|0...0> + e^{i phi} sin(theta)|1...1>."""
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
    """Dataset returning quantum state vectors and target values."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int):
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class QuantumRegressor(nn.Module):
    """Quantum variational circuit with a classical post‑processing head."""
    def __init__(self, num_wires: int, n_layers: int = 2):
        super().__init__()
        self.num_wires = num_wires
        self.n_layers = n_layers

        # PennyLane device
        self.dev = qml.device("default.qubit", wires=num_wires, shots=None)

        # Variational parameters
        self.params = nn.Parameter(pnp.random.randn(n_layers, num_wires, 3))

        # Classical linear head
        self.head = nn.Linear(num_wires, 1)

        # Define quantum node
        def _qcircuit(x, params):
            qml.QubitStateVector(x, wires=range(self.num_wires))
            for layer in range(self.n_layers):
                for w in range(self.num_wires):
                    qml.RX(params[layer, w, 0], wires=w)
                    qml.RY(params[layer, w, 1], wires=w)
                    qml.RZ(params[layer, w, 2], wires=w)
            return [qml.expval(qml.PauliZ(w)) for w in range(self.num_wires)]

        self.qnode = qml.qnode(self.dev, interface="torch")( _qcircuit )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        # Normalize each state vector
        norms = torch.norm(state_batch, dim=1, keepdim=True)
        state_batch = state_batch / norms
        # Compute features for each sample
        features_list = []
        for i in range(bsz):
            x = state_batch[i]
            feat = self.qnode(x, self.params)
            features_list.append(feat)
        features = torch.stack(features_list)
        return self.head(features).squeeze(-1)

__all__ = ["QuantumRegressor", "RegressionDataset", "generate_superposition_data"]
