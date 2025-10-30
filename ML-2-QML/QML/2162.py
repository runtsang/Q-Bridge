"""QuantumHybridClassifier: quantum variant with a variational circuit head.

The module implements a CNN backbone identical to the classical version.
The classification head is a variational quantum circuit that maps a
10‑dimensional feature vector to a single expectation value.  The
circuit uses per‑feature RY rotations followed by a CNOT ring and
measures Pauli‑Z on the first qubit.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml


class QuantumMLPHead(nn.Module):
    """Variational quantum circuit used as the classification head."""

    def __init__(self, n_qubits: int, device: str | None = None) -> None:
        super().__init__()
        if device is None:
            device = "default.qubit"
        self.dev = qml.device(device, wires=n_qubits)
        self.n_qubits = n_qubits
        # Learnable shift applied to each feature
        self.shift = nn.Parameter(torch.zeros(n_qubits))
        self._circuit = qml.qnode(self.dev, interface="torch")(self._run)

    def _run(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, n_qubits]
        for i in range(self.n_qubits):
            qml.RY(x[:, i] + self.shift[i], wires=i)
        # Entangling layer – a simple CNOT ring
        for i in range(self.n_qubits):
            qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
        return qml.expval(qml.PauliZ(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return self._circuit(x)


class QuantumHybridClassifier(nn.Module):
    """CNN followed by a variational quantum circuit for binary classification."""

    def __init__(self) -> None:
        super().__init__()
        # Convolutional backbone identical to the classical version
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(0.2)
        self.drop2 = nn.Dropout2d(0.5)

        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        # 10‑dimensional feature vector to feed the quantum circuit
        self.fc3 = nn.Linear(84, 10)

        self.quantum_head = QuantumMLPHead(n_qubits=10, device="default.qubit")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # shape: [batch, 10]

        logits = self.quantum_head(x).squeeze(-1)
        probs = torch.sigmoid(logits)
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = ["QuantumHybridClassifier"]
