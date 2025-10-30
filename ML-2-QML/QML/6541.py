"""
HybridQuantumCNN – Quantum‑classical hybrid model using Pennylane.

The quantum layer is a parameterised variational circuit that accepts the
features from the classical backbone.  The circuit is built with Pennylane,
leveraging automatic differentiation and allowing the user to swap between
different backends (e.g., default simulator, Aer, or real hardware).
"""

from __future__ import annotations

import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QuantumLayer(nn.Module):
    """Variational quantum layer with entanglement and rotation blocks."""
    def __init__(self, n_qubits: int, n_layers: int = 2, dev_name: str = "default.qubit"):
        super().__init__()
        self.n_qubits = n_qubits
        self.dev = qml.device(dev_name, wires=n_qubits)
        self.n_layers = n_layers
        # Parameters are stored as a learnable torch tensor
        self.params = nn.Parameter(torch.randn(n_layers, n_qubits, 3))

    def _circuit(self, x, params):
        # Encode classical input into rotation angles
        for i in range(self.n_qubits):
            qml.RY(x[i], wires=i)

        # Variational layers
        for layer in range(self.n_layers):
            for qubit in range(self.n_qubits):
                qml.Rot(params[layer, qubit, 0],
                        params[layer, qubit, 1],
                        params[layer, qubit, 2],
                        wires=qubit)
            for qubit in range(self.n_qubits - 1):
                qml.CNOT(wires=[qubit, qubit + 1])

        return qml.expval(qml.PauliZ(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is a 1‑D tensor of length n_qubits
        x_np = x.detach().cpu().numpy()
        expectation = qml.QNode(self._circuit, self.dev, interface="torch")(x_np, self.params)
        return expectation.unsqueeze(0)


class HybridFunction(nn.Module):
    """Bridge between the classical CNN and the quantum layer."""
    def __init__(self, quantum_layer: QuantumLayer, shift: float = 0.0):
        super().__init__()
        self.quantum_layer = quantum_layer
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input shape matches quantum layer expectation
        if x.dim() > 1:
            x = x.view(x.size(0), -1)
        # Pass through quantum circuit
        q_out = self.quantum_layer(x)
        return torch.sigmoid(q_out + self.shift)


class HybridQuantumCNN(nn.Module):
    """ResNet‑style CNN followed by a Pennylane quantum expectation head."""
    def __init__(self, n_qubits: int = 4, shift: float = 0.0):
        super().__init__()
        # Classical backbone (same as ML version for compatibility)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.res1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.res2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, n_qubits)  # output matches number of qubits

        # Quantum head
        self.quantum_layer = QuantumLayer(n_qubits=n_qubits, dev_name="default.qubit")
        self.hybrid = HybridFunction(self.quantum_layer, shift=shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = self.res1(x)
        x = self.res2(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        probs = self.hybrid(x)
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = ["QuantumLayer", "HybridFunction", "HybridQuantumCNN"]
