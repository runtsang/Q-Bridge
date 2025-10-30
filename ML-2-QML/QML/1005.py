"""QuantumHybridBinaryClassifier – quantum counterpart with variational circuit.

This module implements a hybrid model that mirrors the classical
architecture but replaces the dense head with a parameterised
variational quantum circuit executed on a Pennylane device.
Automatic differentiation is enabled via a custom torch.autograd
Function that forwards gradients through the circuit.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pennylane as qml
from pennylane import numpy as pnp

# Define a default qubit device; GPU acceleration is available if the backend supports it
dev = qml.device("default.qubit", wires=2, shots=1024)

class QuantumCircuit:
    """Parameterised two‑qubit variational circuit."""
    def __init__(self):
        # Parameters are stored as a Pennylane numpy array with gradients enabled
        self.params = pnp.array([0.0, 0.0, 0.0, 0.0], requires_grad=True)

    def circuit(self, params):
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        qml.RY(params[0], wires=0)
        qml.RY(params[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RY(params[2], wires=0)
        qml.RY(params[3], wires=1)
        return qml.expval(qml.PauliZ(0))

    def expectation(self, inputs: np.ndarray) -> np.ndarray:
        """Compute expectation for a batch of scalar inputs."""
        out = []
        for x in inputs:
            params = self.params.copy()
            params[0] += x  # inject input as a rotation angle on the first qubit
            out.append(self.circuit(params))
        return np.array(out)

class HybridFunction(torch.autograd.Function):
    """Bridge between torch tensors and the quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit) -> torch.Tensor:
        ctx.circuit = circuit
        ctx.save_for_backward(inputs)
        inputs_np = inputs.detach().cpu().numpy()
        exp = circuit.expectation(inputs_np)
        return torch.from_numpy(exp).to(inputs.device).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, = ctx.saved_tensors
        eps = 1e-5
        inputs_np = inputs.detach().cpu().numpy()
        grad_inputs = []
        for val in inputs_np:
            plus = ctx.circuit.expectation(np.array([val + eps]))[0]
            minus = ctx.circuit.expectation(np.array([val - eps]))[0]
            grad_inputs.append((plus - minus) / (2 * eps))
        grad_inputs = np.array(grad_inputs)
        grad_tensor = torch.from_numpy(grad_inputs).to(grad_output.device).float()
        return grad_tensor * grad_output, None

class Hybrid(nn.Module):
    """Hybrid layer that forwards activations through a variational circuit."""
    def __init__(self):
        super().__init__()
        self.quantum_circuit = QuantumCircuit()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs, self.quantum_circuit)

class QCNet(nn.Module):
    """Hybrid CNN followed by a variational quantum head."""
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.hybrid = Hybrid()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(inputs))
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
        probs = self.hybrid(x)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["QuantumCircuit", "HybridFunction", "Hybrid", "QCNet"]
