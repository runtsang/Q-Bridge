"""Hybrid quantum‑classical CNN for binary classification.

This module replaces the Qiskit‑based circuit with a Pennylane QNode
that supports parameter‑shift gradients and an entangling layer,
while maintaining the same HybridNet interface.
"""

import torch
import torch.nn as nn
import pennylane as qml
import pennylane.numpy as pnp


class QuantumLayer(nn.Module):
    """Quantum expectation layer implemented with Pennylane."""
    def __init__(self, n_qubits: int, device: str = "default.qubit", shots: int = 1024):
        super().__init__()
        self.n_qubits = n_qubits
        self.dev = qml.device(device, wires=n_qubits, shots=shots)

        @qml.qnode(self.dev, interface="torch", diff_method="parameter-shift")
        def circuit(inputs: torch.Tensor):
            for i in range(self.n_qubits):
                qml.RX(inputs[i], wires=i)
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure x has shape (batch, n_qubits)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.circuit(x)


class HybridNet(nn.Module):
    """Convolutional network followed by a Pennylane quantum expectation head."""
    def __init__(self, n_qubits: int = 4, shots: int = 1024):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64, n_qubits, bias=False)
        self.quantum = QuantumLayer(n_qubits, shots=shots)
        self.shift = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc(x)
        q_out = self.quantum(x)
        probs = torch.sigmoid(q_out + self.shift)
        return torch.cat((probs.unsqueeze(-1), 1 - probs.unsqueeze(-1)), dim=-1)


__all__ = ["HybridNet", "QuantumLayer"]
