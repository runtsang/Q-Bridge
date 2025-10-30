"""Hybrid classical‑quantum binary classifier with a Pennylane back‑end.

The quantum circuit is a two‑qubit ansatz that applies a global Hadamard,
a parameterised Ry rotation on each qubit, and a chain of CNOT gates.
The circuit is executed on a Pennylane device with automatic differentiation,
allowing the hybrid layer to be trained end‑to‑end with PyTorch.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import pennylane as qml


class QuantumCircuit:
    """Parameterised two‑qubit circuit executed on a Pennylane device."""

    def __init__(self, n_qubits: int = 2, device_name: str = "default.qubit") -> None:
        self.n_qubits = n_qubits
        self.wires = list(range(n_qubits))
        self.dev = qml.device(device_name, wires=self.wires)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(theta: torch.Tensor) -> torch.Tensor:
            for w in self.wires:
                qml.Hadamard(w)
            for i, w in enumerate(self.wires):
                qml.RY(theta[i], w)
            # entanglement
            for i in range(self.n_qubits - 1):
                qml.CNOT(self.wires[i], self.wires[i + 1])
            return qml.expval(qml.PauliZ(self.wires[0]))

        self.circuit = circuit

    def run(self, thetas: torch.Tensor) -> torch.Tensor:
        """Execute the circuit for a batch of parameter vectors."""
        # thetas shape (batch, n_qubits)
        return torch.stack([self.circuit(thetas[i]) for i in range(thetas.shape[0])]).unsqueeze(-1)


class Hybrid(nn.Module):
    """Hybrid layer that forwards activations through a quantum circuit."""

    def __init__(self, n_qubits: int = 2, device_name: str = "default.qubit") -> None:
        super().__init__()
        self.quantum_circuit = QuantumCircuit(n_qubits, device_name)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs shape (batch, n_qubits)
        return self.quantum_circuit.run(inputs)


class QCNet(nn.Module):
    """Convolutional network followed by a quantum expectation head."""

    def __init__(self) -> None:
        super().__init__()
        # Feature extractor identical to the classical version
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.bn1   = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.bn2   = nn.BatchNorm2d(15)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Classifier
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)   # two logits for the hybrid head

        # Hybrid head
        self.hybrid = Hybrid(n_qubits=2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(inputs)))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)          # shape (batch, 2)
        # Quantum expectation on the first logit (treated as two‑qubit parameters)
        # Map the expectation [-1,1] to a probability [0,1]
        expectation = self.hybrid(x)   # shape (batch, 1)
        prob_class1 = (expectation + 1.0) / 2.0
        return torch.cat((prob_class1, 1 - prob_class1), dim=-1)


__all__ = ["QuantumCircuit", "Hybrid", "QCNet"]
