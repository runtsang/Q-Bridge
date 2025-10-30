"""HybridQCNet – classical backbone with a PennyLane variational quantum head.

This module mirrors the classical implementation but replaces the
Qiskit quantum layer with a PennyLane variational circuit.
The design allows the quantum backend to be swapped out for a
real device or a different simulator.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import numpy as np

# Classical backbone: identical to the ML implementation
class _ResNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=stride, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class _FeaturePyramid(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = _ResNetBlock(3, 16)
        self.block2 = _ResNetBlock(16, 32, stride=2)
        self.block3 = _ResNetBlock(32, 64, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = F.max_pool2d(x, 2)
        x = self.block2(x)
        x = F.max_pool2d(x, 2)
        x = self.block3(x)
        return x

# Quantum layer using PennyLane
class _HybridLayer(nn.Module):
    def __init__(self, n_qubits: int, dev=None, shift: float = np.pi / 2):
        super().__init__()
        self.n_qubits = n_qubits
        self.shift = shift
        if dev is None:
            dev = qml.device("default.qubit", wires=n_qubits, shots=1024)
        self.dev = dev
        # Define the variational circuit
        @qml.qnode(self.dev, interface="torch")
        def circuit(params):
            # Simple entangling circuit
            for i in range(self.n_qubits):
                qml.RY(params[i], wires=i)
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(0))
        self._circuit = circuit

    def set_backend(self, dev):
        """Replace the PennyLane device."""
        self.dev = dev
        @qml.qnode(self.dev, interface="torch")
        def circuit(params):
            for i in range(self.n_qubits):
                qml.RY(params[i], wires=i)
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(0))
        self._circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Map the input to a vector of parameters for the circuit
        # Here we simply repeat the scalar value across all qubits
        params = x.repeat(self.n_qubits, 1).t()  # shape (batch, n_qubits)
        # Compute expectation values
        expectations = self._circuit(params)
        return expectations.unsqueeze(-1)

class HybridQCNet(nn.Module):
    """Full hybrid convolutional‑quantum binary classifier using PennyLane."""

    def __init__(self, n_qubits: int = 2, dev=None, shift: float = np.pi / 2):
        super().__init__()
        self.backbone = _FeaturePyramid()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 120),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(84, 1)
        )
        self.hybrid = _HybridLayer(n_qubits, dev=dev, shift=shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.classifier(features)
        probs = torch.sigmoid(logits)
        q_out = self.hybrid(probs)
        return torch.cat([q_out, 1 - q_out], dim=-1)

    def set_backend(self, dev):
        """Set a new PennyLane device for the hybrid layer."""
        self.hybrid.set_backend(dev)

__all__ = ["HybridQCNet"]
