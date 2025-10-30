"""HybridBinaryClassifier – quantum‑enhanced CNN with a variational head.

The quantum version mirrors the classical architecture but replaces the
final linear layer with a variational quantum circuit that learns a
parameter‑shared weight matrix.  The circuit is implemented with
PennyLane and integrated into PyTorch via a custom autograd Function.

Classes
-------
ResidualBlock : nn.Module
    Re‑used from the classical implementation.
VariationalQuantumLayer : nn.Module
    A parameter‑shared 2‑qubit variational circuit.
HybridBinaryClassifier : nn.Module
    Full model that couples the CNN backbone with the quantum head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import numpy as np


class ResidualBlock(nn.Module):
    """3×3 convolutional residual block with optional down‑sampling."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if in_channels!= out_channels or stride!= 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return self.relu(out + self.shortcut(x))


class VariationalQuantumLayer(nn.Module):
    """Parameter‑shared variational circuit that outputs an expectation value."""
    def __init__(self, n_qubits: int = 2, dev: qml.Device | None = None):
        super().__init__()
        self.n_qubits = n_qubits
        self.dev = dev or qml.device("default.qubit", wires=n_qubits)

        # Learnable parameters – one rotation per qubit
        self.params = nn.Parameter(torch.randn(n_qubits))

        @qml.qnode(self.dev, interface="torch")
        def circuit(x, params):
            for i in range(self.n_qubits):
                qml.RY(x, wires=i)
                qml.RZ(params[i], wires=i)
            # Entangle all qubits
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accepts a scalar per sample; batch processing handled externally
        return self.circuit(x, self.params)


class HybridBinaryClassifier(nn.Module):
    """CNN backbone with a residual block and a variational quantum head."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.2),
            ResidualBlock(6, 15, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.5),
        )

        dummy = torch.zeros(1, 3, 32, 32)
        feat_size = self.features(dummy).view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Linear(feat_size, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, 1),
        )
        self.quantum_head = VariationalQuantumLayer(n_qubits=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        # Quantum head expects scalar inputs; process each sample individually
        q_out = []
        for val in x.squeeze():
            q_out.append(self.quantum_head(val))
        q_out = torch.stack(q_out)
        probs = torch.sigmoid(q_out)
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = ["ResidualBlock", "VariationalQuantumLayer", "HybridBinaryClassifier"]
