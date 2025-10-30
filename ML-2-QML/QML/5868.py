"""Quantum‑classical hybrid network for binary classification using Qiskit.

This module extends the original hybrid architecture by replacing the
parameterised circuit with a more expressive variational circuit that
includes entanglement and a parameter‑shift rule for gradient
estimation.  The quantum head is implemented as a differentiable
torch.autograd.Function that forwards activations through the circuit.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.providers.aer import AerSimulator


class QuantumVariationalCircuit:
    """Parameterized 2‑qubit circuit with entanglement."""

    def __init__(self, backend: qiskit.providers.Backend, shots: int = 1024) -> None:
        self.backend = backend
        self.shots = shots
        self.circuit = QuantumCircuit(2)
        self.theta = qiskit.circuit.Parameter("theta")
        # Simple entangling circuit
        self.circuit.h(0)
        self.circuit.cx(0, 1)
        self.circuit.ry(self.theta, 0)
        self.circuit.ry(self.theta, 1)
        self.circuit.measure_all()

    def run(self, theta: float) -> float:
        """Return expectation value of Z on qubit 0 for a given theta."""
        bound = {self.theta: theta}
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(compiled, shots=self.shots, parameter_binds=[bound])
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        # Convert counts to expectation value of Z
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        probs = counts / self.shots
        return np.sum(states * probs)


class QuantumHybridFunction(torch.autograd.Function):
    """Differentiable interface between PyTorch and the quantum circuit."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumVariationalCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        ctx.save_for_backward(inputs)
        theta_vals = inputs.squeeze().tolist()
        exp_vals = np.array([circuit.run(theta) for theta in theta_vals])
        return torch.sigmoid(torch.tensor(exp_vals, dtype=torch.float32).unsqueeze(-1))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, = ctx.saved_tensors
        shift = ctx.shift
        circuit = ctx.circuit
        grad_inputs = []
        for theta in inputs.squeeze().tolist():
            right = circuit.run(theta + shift)
            left = circuit.run(theta - shift)
            grad_inputs.append((right - left) / 2.0)
        grad_inputs = torch.tensor(grad_inputs, dtype=torch.float32).unsqueeze(-1)
        return grad_inputs * grad_output, None, None


class QuantumHybridHead(nn.Module):
    """Hybrid layer that forwards activations through a quantum circuit."""

    def __init__(self, shift: float = np.pi / 2) -> None:
        super().__init__()
        self.shift = shift
        self.circuit = QuantumVariationalCircuit(AerSimulator(), shots=1024)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return QuantumHybridFunction.apply(x, self.circuit, self.shift)


class HybridQCNet(nn.Module):
    """Quantum‑classical hybrid network identical to the classical version
    but with a quantum head."""

    def __init__(self, shift: float = np.pi / 2) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(0.2)
        self.drop2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.head = QuantumHybridHead(shift=shift)

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
        probs = self.head(x)
        return torch.cat((probs, 1 - probs), dim=-1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        probs = self.forward(x)
        return torch.argmax(probs, dim=-1)


__all__ = ["HybridQCNet"]
