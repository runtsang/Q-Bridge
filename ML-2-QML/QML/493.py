"""Hybrid quantum-classical model using Pennylane, inspired by QuantumNAT."""

import torch
import torch.nn as nn
import pennylane as qml
import pennylane.numpy as np


class QuantumNatModel(nn.Module):
    """Hybrid model that encodes classical features into a 4‑qubit variational circuit."""

    def __init__(self, n_qubits: int = 4, n_layers: int = 2) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Classical feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Pennylane device and circuit
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")

        # Classical readout head
        self.fc = nn.Linear(n_qubits, 4)
        self.norm = nn.BatchNorm1d(4)

    def _circuit(self, x: torch.Tensor) -> torch.Tensor:
        """Parameter‑ized quantum circuit."""
        # Encode input features into rotation angles
        for i in range(self.n_qubits):
            qml.RY(x[i], wires=i)

        # Entangling layers
        for _ in range(self.n_layers):
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            for i in range(self.n_qubits):
                qml.RZ(torch.randn(1).item(), wires=i)

        # Measurement
        return torch.stack([qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        feat = self.features(x)
        flat = feat.view(bsz, -1)
        # Reduce to n_qubits via mean over spatial dimensions
        encode = torch.mean(flat, dim=1, keepdim=True)
        # Broadcast to each qubit
        encoded = encode.repeat(1, self.n_qubits)
        # Execute quantum circuit for each sample
        q_out = []
        for sample in encoded:
            q_out.append(self.qnode(sample))
        q_out = torch.stack(q_out)
        out = self.fc(q_out)
        return self.norm(out)


__all__ = ["QuantumNatModel"]
