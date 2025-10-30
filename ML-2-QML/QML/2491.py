"""Hybrid quantum-classical neural network for binary classification.

This module implements a convolutional feature extractor followed by a
parameterised quantum circuit head. The quantum circuit is built with Qiskit
and includes clipping of rotation angles for numerical stability. The
output of the circuit is passed through a scaling and shift transform
to match the probability output of the classical head. Gradients are
computed using the parameter‑shift rule, ensuring differentiability
within PyTorch.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import qiskit
from qiskit import assemble, transpile

@dataclass
class ScaleShift:
    """Scaling and shifting applied to the quantum expectation."""
    scale: float = 1.0
    shift: float = 0.0

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

class QuantumCircuit:
    """Two‑qubit circuit with H, Ry and measurement, angles clipped for stability."""

    def __init__(self, n_qubits: int, backend, shots: int, angle_bound: float = 5.0) -> None:
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        all_qubits = list(range(n_qubits))
        self.theta = qiskit.circuit.Parameter("theta")
        self._circuit.h(all_qubits)
        self._circuit.barrier()
        self._circuit.ry(self.theta, all_qubits)
        self._circuit.measure_all()
        self.backend = backend
        self.shots = shots
        self.angle_bound = angle_bound

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """Execute the circuit for a list of angles, returning expectation."""
        clipped = np.clip(thetas, -self.angle_bound, self.angle_bound)
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in clipped],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()

        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probabilities = counts / self.shots
            return np.sum(states * probabilities)

        if isinstance(result, list):
            return np.array([expectation(item) for item in result])
        return np.array([expectation(result)])

class HybridFunction(torch.autograd.Function):
    """Differentiable interface that runs the quantum circuit and applies shift/scale."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float, scale: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.scale = scale
        ctx.quantum_circuit = circuit
        expectation = ctx.quantum_circuit.run(inputs.tolist())
        result = torch.tensor(expectation, dtype=torch.float32)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = np.ones_like(inputs.numpy()) * ctx.shift
        gradients = []
        for idx, value in enumerate(inputs.numpy()):
            exp_plus = ctx.quantum_circuit.run([value + shift[idx]])[0]
            exp_minus = ctx.quantum_circuit.run([value - shift[idx]])[0]
            gradients.append(exp_plus - exp_minus)
        gradients = torch.tensor(gradients, dtype=torch.float32)
        return gradients * ctx.scale * grad_output, None, None, None

class Hybrid(nn.Module):
    """Quantum head that maps a scalar to a probability."""

    def __init__(self, n_qubits: int, backend, shots: int, shift: float, scale: float) -> None:
        super().__init__()
        self.quantum_circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift
        self.scale = scale

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        squeezed = torch.squeeze(inputs) if inputs.shape!= torch.Size([1, 1]) else inputs[0]
        return HybridFunction.apply(squeezed, self.quantum_circuit, self.shift, self.scale)

class HybridQCNet(nn.Module):
    """Convolutional network followed by a quantum expectation head."""

    def __init__(self, backend: qiskit.providers.Backend, shots: int = 100, shift: float = np.pi / 2, scale: float = 1.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.hybrid = Hybrid(self.fc3.out_features, backend, shots, shift, scale)

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

__all__ = ["ScaleShift", "QuantumCircuit", "HybridFunction", "Hybrid", "HybridQCNet"]
