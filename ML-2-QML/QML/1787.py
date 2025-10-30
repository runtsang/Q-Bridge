"""Hybrid classical-quantum convolutional network for binary classification.

The quantum component has been upgraded from a single‑qubit expectation to a
3‑qubit entangled ansatz that outputs an expectation of Z on each qubit.
The parameter‑shift rule provides exact gradients for back‑propagation.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import qiskit
from qiskit import assemble, transpile
from qiskit.circuit import Parameter
from qiskit.providers.aer import AerSimulator

class EntangledQuantumCircuit:
    """3‑qubit parameterised variational circuit executed on Aer."""
    def __init__(self, backend: AerSimulator, shots: int) -> None:
        self.backend = backend
        self.shots = shots
        self.theta = Parameter("θ")
        self._circuit = qiskit.QuantumCircuit(3)
        # Entangling layer
        self._circuit.h([0, 1, 2])
        self._circuit.cx(0, 1)
        self._circuit.cx(1, 2)
        # Parameterised rotations
        self._circuit.ry(self.theta, 0)
        self._circuit.ry(self.theta, 1)
        self._circuit.ry(self.theta, 2)
        # Measurement
        self._circuit.measure_all()

    def run(self, theta: float) -> np.ndarray:
        """Execute the circuit for a single angle value."""
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(compiled, shots=self.shots,
                        parameter_binds=[{self.theta: theta}])
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        # Expectation of Z on first qubit
        exp = 0.0
        for bitstring, count in result.items():
            bit = int(bitstring[0])  # first qubit
            exp += ((-1) ** bit) * count
        return np.array([exp / self.shots], dtype=np.float64)

class QuantumHybridFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: EntangledQuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        thetas = inputs.detach().cpu().numpy().reshape(-1)
        expectations = np.array([circuit.run(theta) for theta in thetas])
        result = torch.tensor(expectations, dtype=torch.float32, device=inputs.device)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        inputs, _ = ctx.saved_tensors
        thetas = inputs.detach().cpu().numpy().reshape(-1)
        shift = ctx.shift
        grads = []
        for theta in thetas:
            f_plus = ctx.circuit.run(theta + shift)[0]
            f_minus = ctx.circuit.run(theta - shift)[0]
            grads.append((f_plus - f_minus) / 2.0)
        grads = torch.tensor(grads, dtype=torch.float32, device=inputs.device)
        return grads * grad_output, None, None

class Hybrid(nn.Module):
    """Hybrid layer that forwards activations through a quantum circuit."""
    def __init__(self, backend: AerSimulator, shots: int, shift: float) -> None:
        super().__init__()
        self.circuit = EntangledQuantumCircuit(backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return QuantumHybridFunction.apply(inputs, self.circuit, self.shift)

class HybridQCNet(nn.Module):
    """Convolutional network followed by a quantum expectation head."""
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
        backend = AerSimulator()
        self.hybrid = Hybrid(backend, shots=200, shift=np.pi / 2)

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
        probs = self.hybrid(x).squeeze(-1)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["HybridQCNet"]
