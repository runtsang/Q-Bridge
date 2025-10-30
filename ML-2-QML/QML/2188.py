"""Hybrid classical‑quantum binary classifier using Pennylane.

The quantum head is implemented as a 3‑qubit entangling circuit
with a parameter‑shift derivative.  The rest of the network mirrors the
classical architecture but with an additional feature‑wise hybrid layer.
"""

from __future__ import annotations

import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Default qubit simulator device
dev = qml.device("default.qubit", wires=3)


class HybridFunction(torch.autograd.Function):
    """Wrapper that forwards a tensor through a Pennylane QNode.

    The backward pass is handled automatically by Pennylane's autograd
    interface, which implements the parameter‑shift rule.
    """
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: qml.QNode):
        ctx.circuit = circuit
        # Forward pass through the QNode (expects a scalar input)
        return circuit(inputs)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Automatic differentiation provided by Pennylane handles gradients.
        return None, None


class Hybrid(nn.Module):
    """Hybrid layer for the quantum expectation head."""

    def __init__(self, n_qubits: int = 3, shift: float = np.pi / 2):
        super().__init__()
        self.shift = shift
        # QNode that maps a scalar parameter to a Pauli‑Z expectation
        @qml.qnode(dev, interface="torch")
        def circuit(theta: torch.Tensor):
            for i in range(n_qubits):
                qml.RY(theta, wires=i)
            # Entangling layer
            qml.CZ(wires=[0, 1])
            qml.CZ(wires=[1, 2])
            qml.CZ(wires=[0, 2])
            return qml.expval(qml.PauliZ(0))
        self.circuit = circuit

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs is expected to be of shape (batch_size, 1)
        # Pennylane QNodes are not vectorized over batch, so we loop.
        outputs = []
        for x in inputs.squeeze():
            outputs.append(self.circuit(x))
        return torch.stack(outputs).unsqueeze(-1)  # shape (batch_size, 1)


class HybridQCNet(nn.Module):
    """Convolutional net followed by a Pennylane quantum hybrid head."""

    def __init__(self, n_qubits: int = 3, shift: float = np.pi / 2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(15)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Flattened size for 32x32 input after conv layers
        self._flattened = 15 * 6 * 6  # 540
        self.fc1 = nn.Linear(self._flattened, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        self.hybrid = Hybrid(n_qubits, shift)

    def _conv_block(self, x, conv, bn):
        out = F.relu(bn(conv(x)))
        out = self.pool(out)
        out = self.drop1(out)
        return out

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out = self._conv_block(inputs, self.conv1, self.bn1)
        out = self._conv_block(out, self.conv2, self.bn2)
        out = torch.flatten(out, 1)
        out = F.relu(self.fc1(out))
        out = self.drop2(out)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        probs = self.hybrid(out)
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = ["HybridQCNet", "Hybrid", "HybridFunction"]
