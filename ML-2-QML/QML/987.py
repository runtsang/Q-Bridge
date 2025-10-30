"""HybridQuantumBinaryClassifier – Quantum hybrid model with a parameter‑shifted expectation head."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import qiskit
from qiskit import assemble, transpile
from qiskit.circuit import Parameter
from qiskit.providers.aer import AerSimulator

class ParametrizedQuantumCircuit:
    """A flexible two‑qubit circuit with configurable depth."""
    def __init__(self, depth: int, backend=None, shots: int = 1024):
        self.depth = depth
        self.backend = backend or AerSimulator()
        self.shots = shots
        self.theta = Parameter("θ")
        self._build_circuit()

    def _build_circuit(self) -> None:
        self.circuit = qiskit.QuantumCircuit(2)
        for _ in range(self.depth):
            self.circuit.h([0, 1])
            self.circuit.cx(0, 1)
            self.circuit.ry(self.theta, 0)
            self.circuit.ry(self.theta, 1)
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        def expectation(count_dict):
            probs = {int(k, 2): v / self.shots for k, v in count_dict.items()}
            return probs.get(0, 0) - probs.get(3, 0)
        if isinstance(result, list):
            return np.array([expectation(r) for r in result])
        return np.array([expectation(result)])

class HybridFunction(torch.autograd.Function):
    """Differentiable interface to the quantum circuit via parameter‑shift."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: ParametrizedQuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        expectation = circuit.run(inputs.tolist())
        result = torch.tensor(expectation, dtype=torch.float32)
        ctx.save_for_backward(inputs)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, = ctx.saved_tensors
        shift = ctx.shift
        grads = []
        for val in inputs.tolist():
            right = ctx.circuit.run([val + shift])[0]
            left = ctx.circuit.run([val - shift])[0]
            grads.append(right - left)
        grads = torch.tensor(grads, dtype=torch.float32)
        return grads * grad_output, None, None

class HybridQuantumBinaryClassifier(nn.Module):
    """Quantum hybrid classifier with a parameter‑shifted expectation head."""
    def __init__(self, depth: int = 3, shift: float = np.pi / 2, shots: int = 1024):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        self.quantum_circuit = ParametrizedQuantumCircuit(depth, shots=shots)
        self.shift = shift

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
        x = HybridFunction.apply(x.squeeze(), self.quantum_circuit, self.shift)
        probs = torch.sigmoid(x)
        return torch.cat((probs.unsqueeze(-1), (1 - probs).unsqueeze(-1)), dim=-1)

__all__ = ["HybridQuantumBinaryClassifier"]
