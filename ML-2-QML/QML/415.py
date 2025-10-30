"""HybridClassifier: Quantum hybrid implementation using Pennylane.

This module extends the original hybrid quantum classifier by replacing the
single‑qubit expectation head with a 4‑qubit variational circuit that
encodes the classical features via a rotation‑based feature map.  The
circuit is fully differentiable through Pennylane's autograd
integration and can be trained together with the classical backbone.
"""

import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QuantumCircuit(nn.Module):
    """Variational quantum circuit with a rotation feature map."""

    def __init__(self, n_qubits: int, n_layers: int = 2) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.params = nn.Parameter(torch.randn(n_layers, n_qubits, 3))

    def circuit(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit_fn(x, params):
            for i in range(self.n_qubits):
                qml.RY(x[i], wires=i)
            for layer in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.Rot(
                        params[layer, i, 0],
                        params[layer, i, 1],
                        params[layer, i, 2],
                        wires=i,
                    )
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(0))
        return circuit_fn(x, params)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch = inputs.shape[0]
        outputs = []
        for i in range(batch):
            out = self.circuit(inputs[i], self.params)
            outputs.append(out)
        return torch.stack(outputs)


class HybridClassifier(nn.Module):
    """Convolutional network followed by a variational quantum head."""

    def __init__(self, n_qubits: int = 4) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        self.quantum = QuantumCircuit(n_qubits=n_qubits, n_layers=2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(inputs))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x).squeeze(-1)  # shape: (batch,)
        q_input = x.unsqueeze(1).repeat(1, self.quantum.n_qubits)  # shape: (batch, n_qubits)
        q_out = self.quantum(q_input)  # shape: (batch,)
        probs = 0.5 * (q_out + 1.0)
        return torch.stack([probs, 1 - probs], dim=-1)


__all__ = ["QuantumCircuit", "HybridClassifier"]
