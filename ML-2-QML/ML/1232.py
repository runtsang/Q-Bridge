"""HybridQuantumNet: Classical backbone with a quantum kernel head.

This module defines a PyTorch neural network that keeps the original
CNN architecture but replaces the final dense head with a
trainable quantum kernel implemented via PennyLane.  The quantum
kernel is wrapped in a custom autograd function so that gradients
flow seamlessly through the quantum circuit.  Batch‑normalization
and dropout are added around the quantum head to improve regularization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import numpy as np

class QuantumKernel(nn.Module):
    def __init__(self, n_qubits: int, wires: list[int], shift: float = np.pi / 2):
        super().__init__()
        self.n_qubits = n_qubits
        self.shift = shift
        self.wires = wires
        self.dev = qml.device("default.qubit", wires=wires)
        # Variational parameters
        self.params = nn.Parameter(torch.randn(2 * n_qubits))
        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs, params):
            for i in range(self.n_qubits):
                qml.RY(inputs[i], wires=wires[i])
            for i in range(self.n_qubits):
                qml.RZ(params[i], wires=wires[i])
            for i in range(self.n_qubits):
                qml.RX(params[self.n_qubits + i], wires=wires[i])
            return qml.expval(qml.PauliZ(wires[0]))
        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expecting x shape (batch, n_features) where n_features >= n_qubits
        angles = x[:, :self.n_qubits]
        return self.circuit(angles, self.params)

class QuantumLayerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, kernel: QuantumKernel, shift: float):
        ctx.shift = shift
        ctx.kernel = kernel
        outputs = kernel(inputs)
        ctx.save_for_backward(inputs, outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, outputs = ctx.saved_tensors
        shift = ctx.shift
        grads = []
        for i in range(inputs.shape[1]):
            plus = inputs.clone()
            minus = inputs.clone()
            plus[:, i] += shift
            minus[:, i] -= shift
            out_plus = ctx.kernel(plus)
            out_minus = ctx.kernel(minus)
            grads.append((out_plus - out_minus) / (2 * shift))
        grad_inputs = torch.stack(grads, dim=1)
        return grad_inputs * grad_output.unsqueeze(1), None, None

class HybridQuantumNet(nn.Module):
    """CNN backbone followed by a quantum kernel head."""
    def __init__(self):
        super().__init__()
        # Classical backbone identical to original
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        # Quantum kernel head
        self.quantum = QuantumKernel(n_qubits=1, wires=[0], shift=np.pi / 2)
        self.bn = nn.BatchNorm1d(1)
        self.dropout = nn.Dropout(p=0.3)

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
        q_out = QuantumLayerFunction.apply(x, self.quantum, self.quantum.shift)
        q_out = self.bn(q_out)
        q_out = self.dropout(q_out)
        probs = torch.sigmoid(q_out)
        return torch.cat([probs, 1 - probs], dim=-1)
