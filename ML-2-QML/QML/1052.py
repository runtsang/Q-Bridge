"""Quantum‑enhanced hybrid network using Pennylane.

The implementation replaces the dense hybrid head with a parameterised
four‑qubit variational circuit.  The circuit is executed on the
``default.qubit`` device and the expectation value of Pauli‑Z is used
as the output of the hybrid layer.  The module keeps the same public
interface as the classical counterpart.
"""

import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QuantumCircuit(nn.Module):
    """Four‑qubit variational circuit with entanglement."""
    def __init__(self, n_qubits: int = 4, dev: qml.Device | None = None) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.dev = dev or qml.device("default.qubit", wires=n_qubits)
        self.ansatz = self._build_ansatz()

    def _build_ansatz(self):
        @qml.qnode(self.dev, interface="torch")
        def circuit(params: torch.Tensor, x: torch.Tensor):
            # Encode input feature into rotation angles
            for i in range(self.n_qubits):
                qml.RY(x[i], wires=i)
            # Variational layers
            for layer in range(2):
                for i in range(self.n_qubits):
                    qml.RY(params[layer, i], wires=i)
                # Entangling layer
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[self.n_qubits - 1, 0])
            return qml.expval(qml.PauliZ(0))
        return circuit

    def forward(self, params: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.ansatz(params, x)

class HybridFunction(nn.Module):
    """Wrapper that applies the variational circuit and a sigmoid."""
    def __init__(self, shift: float = np.pi / 2) -> None:
        super().__init__()
        self.shift = shift
        self.params = nn.Parameter(torch.randn(2, 4) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input has the same dimension as the circuit
        if x.ndim == 1:
            x = x.unsqueeze(0)
        # Pad or truncate to match n_qubits
        n_qubits = 4
        x_adj = torch.zeros((x.size(0), n_qubits), device=x.device)
        x_adj[:, :x.size(1)] = x
        # Compute expectation
        qc = QuantumCircuit(n_qubits)
        exp = qc(self.params, x_adj)
        # Apply shift rule for gradients
        return torch.sigmoid(exp + self.shift)

class Hybrid(nn.Module):
    """Hybrid head that forwards activations through the quantum circuit."""
    def __init__(self, shift: float = np.pi / 2) -> None:
        super().__init__()
        self.func = HybridFunction(shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.func(x)

class ResidualBlock(nn.Module):
    """Simple residual block with two 3×3 convolutions."""
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class HybridNet(nn.Module):
    """CNN backbone identical to the classical version followed by a quantum head."""
    def __init__(self) -> None:
        super().__init__()
        # Backbone (same as classical)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2)
        self.bn1   = nn.BatchNorm2d(32)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.res   = ResidualBlock(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2   = nn.BatchNorm2d(64)

        # MLP head
        self.fc1   = nn.Linear(64 * 4 * 4, 256)
        self.fc2   = nn.Linear(256, 128)

        # Quantum hybrid head
        self.hybrid = Hybrid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.res(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.hybrid(x)
        return torch.cat((logits, 1 - logits), dim=-1)

__all__ = ["QuantumCircuit", "HybridFunction", "Hybrid", "HybridNet"]
