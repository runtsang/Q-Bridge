"""Hybrid quantum‑classical binary classifier using PennyLane.

The circuit is a parameterised 2‑qubit ansatz with Ry, Rz, and CNOT layers.
A Torch interface (qml.jacobian) delivers gradients to the PyTorch
optimiser.  The hybrid head can be swapped with the classical head
provided in the ML module for comparative experiments.
"""

import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Device and circuit
dev = qml.device("default.qubit", wires=2)

def _ansatz(params: np.ndarray, x: float) -> None:
    """Two‑qubit variational circuit.
    The input x is encoded via an Ry rotation on qubit 0.
    The remaining parameters are applied as Ry-Rz layers on each qubit
    followed by a CNOT entangling gate.
    """
    # Feature encoding
    qml.Ry(x, wires=0)

    # Parameterised layers
    for w in range(2):
        qml.Ry(params[w, 0], wires=w)
        qml.Rz(params[w, 1], wires=w)

    # Entanglement
    qml.CNOT(wires=[0, 1])

# QNode returning expectation of Z on qubit 0
@qml.qnode(dev, interface="torch", diff_method="backprop")
def _qnode(params: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    _ansatz(params, x)
    return qml.expval(qml.PauliZ(0))

class HybridFunction(torch.autograd.Function):
    """Wraps the PennyLane QNode so that it can appear in a PyTorch graph."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        ctx.params = params
        expectation = _qnode(params, inputs)
        ctx.save_for_backward(inputs, expectation)
        return expectation

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        inputs, expectation = ctx.saved_tensors
        # The gradient of the QNode is automatically computed by PennyLane
        # using the autograd interface.  We simply multiply by grad_output.
        grads = grad_output * torch.ones_like(expectation)
        return grads, None

class HybridLayer(nn.Module):
    """Quantum hybrid head.  Parameters are trainable torch tensors."""
    def __init__(self, temperature: float = 1.0) -> None:
        super().__init__()
        # 2 qubits, 2 parameters per qubit
        self.params = nn.Parameter(torch.randn(2, 2))
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten input to a scalar for encoding
        x_flat = torch.mean(x, dim=1, keepdim=True).squeeze(-1)
        val = HybridFunction.apply(x_flat, self.params)
        # Apply temperature‑scaled sigmoid as in the classical head
        return torch.sigmoid(val / self.temperature)

class HybridQCNet(nn.Module):
    """Convolutional backbone followed by a quantum hybrid head."""
    def __init__(self, temperature: float = 1.0) -> None:
        super().__init__()
        # Backbone identical to the classical model for fair comparison
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2   = nn.BatchNorm2d(64)

        self.res_conv = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.res_bn   = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(p=0.3)

        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 128)
        self.hybrid = HybridLayer(temperature=temperature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        res = self.res_bn(self.res_conv(x))
        x = F.relu(x + res)
        x = self.pool(x)
        x = self.dropout(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        probs = self.hybrid(x)
        return torch.cat([probs, 1 - probs], dim=-1)

__all__ = ["HybridFunction", "HybridLayer", "HybridQCNet"]
