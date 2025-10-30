"""Quantum‑enhanced hybrid network for binary classification.

The implementation uses PennyLane for efficient circuit construction
and automatic differentiation.  The quantum head is fully
differentiable and can be interchanged with the classical head
defined in the companion module.
"""

import pennylane as qml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumCircuit:
    """Two‑qubit parameterised circuit with entanglement."""
    def __init__(self, n_qubits: int = 2, device: str = "default.qubit", shots: int = 1024):
        self.n_qubits = n_qubits
        self.device = qml.device(device, wires=n_qubits, shots=shots)
        self._build_circuit()

    def _build_circuit(self):
        @qml.qnode(self.device, interface="torch")
        def circuit(params):
            # Global H
            for w in range(self.n_qubits):
                qml.Hadamard(wires=w)
            # Parameterised Ry
            for w, p in enumerate(params):
                qml.RY(p, wires=w)
            # Entanglement
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 0])
            # Measurement
            return qml.expval(qml.PauliZ(0))
        self.circuit = circuit

    def expectation(self, params: torch.Tensor) -> torch.Tensor:
        return self.circuit(params)

class QuantumHybridHead(nn.Module):
    """Hybrid head that forwards activations through a quantum circuit."""
    def __init__(self, n_qubits: int = 2, device: str = "default.qubit", shots: int = 1024, shift: float = np.pi / 2):
        super().__init__()
        self.circuit = QuantumCircuit(n_qubits, device, shots)
        self.shift = shift
        self.shift_param = nn.Parameter(torch.tensor(shift, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is of shape (batch, 1)
        params = x.squeeze(-1) + self.shift_param
        probs = self.circuit.expectation(params)
        return probs.unsqueeze(-1)

class HybridNet(nn.Module):
    """Convolutional network followed by a quantum hybrid head."""
    def __init__(self, dropout: float = 0.5) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=dropout)

        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.head = QuantumHybridHead()

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
        x = self.fc3(x)
        probs = self.head(x)
        return torch.cat([probs, 1 - probs], dim=-1)

__all__ = ["QuantumCircuit", "QuantumHybridHead", "HybridNet"]
