"""Quantum regression model using Pennylane variational circuits."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from torch.utils.data import Dataset


def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate amplitude‑encoded states of the form
    cos(theta)|0…0> + e^{i phi} sin(theta)|1…1>.
    """
    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * np.eye(1, 2 ** num_wires)[0] + np.exp(1j * phis[i]) * np.sin(thetas[i]) * np.eye(1, 2 ** num_wires)[-1]
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels.astype(np.float32)


class RegressionDataset(Dataset):
    """Torch dataset for quantum regression."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)
        self.states = torch.tensor(self.states, dtype=torch.cfloat)
        self.labels = torch.tensor(self.labels, dtype=torch.float32)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": self.states[index],
            "target": self.labels[index],
        }


class QModel(nn.Module):
    """Hybrid quantum‑classical regression model built with Pennylane."""
    def __init__(self, num_wires: int, num_layers: int = 3):
        super().__init__()
        self.num_wires = num_wires
        self.num_layers = num_layers
        self.dev = qml.device("default.qubit", wires=num_wires)

        # Parameter shape: (num_layers, num_wires, 2)
        self.weights = nn.Parameter(torch.randn(num_layers, num_wires, 2))

        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(num_wires))
            for layer in range(num_layers):
                for wire in range(num_wires):
                    qml.RX(weights[layer, wire, 0], wires=wire)
                    qml.RY(weights[layer, wire, 1], wires=wire)
                if layer < num_layers - 1:
                    for wire in range(num_wires - 1):
                        qml.CNOT(wires=[wire, wire + 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(num_wires)]

        self.q_func = qml.QNode(circuit, self.dev, interface="torch")
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        state_batch: batch of amplitude‑encoded state vectors (shape: [batch, 2**num_wires]).
        """
        batch_size = state_batch.shape[0]
        outputs = []
        for i in range(batch_size):
            state = state_batch[i]
            # Use the real part of the first `num_wires` amplitudes as classical inputs
            inputs = torch.real(state[:self.num_wires])
            out = self.q_func(inputs, self.weights)
            outputs.append(out)
        features = torch.stack(outputs)
        return self.head(features).squeeze(-1)


__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
