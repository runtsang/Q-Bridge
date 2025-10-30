"""Quantum‑enhanced QCNet using Pennylane.

The quantum head is a parametrised two‑qubit circuit with entanglement
and a central‑difference shift rule for analytical gradients.  The
class exposes a PyTorch‑compatible hybrid layer that can be dropped
into the classical backbone defined in the ML counterpart.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumCircuit:
    """Two‑qubit variational circuit with entanglement."""
    def __init__(self, n_qubits: int = 2, dev: str = "default.qubit"):
        self.n_qubits = n_qubits
        self.dev = qml.device(dev, wires=n_qubits)
        self.theta = nn.Parameter(torch.randn(n_qubits))

    def _qnode(self, params: np.ndarray) -> float:
        @qml.qnode(self.dev, interface="torch", diff_method="central")
        def circuit(theta: torch.Tensor) -> torch.Tensor:
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
            qml.CNOT(wires=[0, 1])
            qml.RY(theta[0], wires=0)
            qml.RY(theta[1], wires=1)
            return qml.expval(qml.PauliZ(0))

        return circuit(torch.tensor(params))

    def expectation(self, params: np.ndarray) -> float:
        return self._qnode(params).item()


class HybridFunction(torch.autograd.Function):
    """Torch autograd wrapper around the Pennylane qnode."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        expectation = circuit.expectation(inputs.detach().cpu().numpy())
        return torch.tensor([expectation], dtype=torch.float32, device=inputs.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input_val = ctx.circuit.theta.detach().cpu().numpy()
        shift = np.full_like(input_val, ctx.shift)
        grads = []
        for i in range(len(input_val)):
            eps = shift[i]
            right = ctx.circuit.expectation(input_val + eps * np.eye(1, len(input_val), i)[0])
            left = ctx.circuit.expectation(input_val - eps * np.eye(1, len(input_val), i)[0])
            grads.append((right - left) / (2 * eps))
        grad_tensor = torch.tensor(grads, dtype=torch.float32, device=grad_output.device)
        return grad_tensor * grad_output, None, None


class Hybrid(nn.Module):
    """Hybrid layer that forwards activations through a Pennylane circuit."""
    def __init__(self, n_qubits: int = 2, shift: float = np.pi / 2):
        super().__init__()
        self.circuit = QuantumCircuit(n_qubits)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs.squeeze(), self.circuit, self.shift)


class QCNet(nn.Module):
    """Convolutional backbone with a Pennylane quantum expectation head."""
    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)

        self.res1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
        self.res2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
        )

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

        self.hybrid = Hybrid(n_qubits=2, shift=np.pi / 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.flatten(x)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        logits = self.fc2(x)
        quantum_out = self.hybrid(logits)
        probs = torch.sigmoid(quantum_out)
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = ["QCNet", "Hybrid", "HybridFunction", "QuantumCircuit"]
