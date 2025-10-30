"""Hybrid quantum‑classical binary classifier mirroring the classical counterpart."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Iterable, List

import qiskit
from qiskit import assemble, transpile

@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

def _clip(value: float, bound: float = 5.0) -> float:
    return max(-bound, min(bound, value))

class QuantumCircuit:
    """Parameterised two‑qubit variational circuit with clipped parameters."""
    def __init__(self, n_qubits: int, backend, shots: int) -> None:
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.Parameter("theta")

        # Build a single layer template
        self.circuit.h(range(n_qubits))
        self.circuit.barrier()
        self.circuit.ry(self.theta, range(n_qubits))
        if n_qubits == 2:
            self.circuit.cx(0, 1)
        self.circuit.barrier()
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        clipped = np.clip(thetas, -5.0, 5.0)
        compiled = transpile(self.circuit, self.backend)
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
            probs = counts / self.shots
            return np.sum(states * probs)

        if isinstance(result, list):
            return np.array([expectation(item) for item in result])
        return np.array([expectation(result)])

class HybridFunction(torch.autograd.Function):
    """Differentiable bridge between PyTorch and the quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        theta_values = inputs.detach().cpu().numpy().flatten()
        expectation = ctx.circuit.run(theta_values)
        out = torch.tensor(expectation, dtype=torch.float32, device=inputs.device)
        ctx.save_for_backward(inputs, out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        grads = []
        for val in inputs.detach().cpu().numpy().flatten():
            right = ctx.circuit.run([val + shift])
            left = ctx.circuit.run([val - shift])
            grads.append(right - left)
        grad_tensor = torch.tensor(grads, dtype=torch.float32, device=inputs.device)
        return grad_tensor * grad_output, None, None

class Hybrid(nn.Module):
    """Quantum hybrid head that forwards activations through a variational circuit."""
    def __init__(self, n_qubits: int, backend, shots: int, shift: float) -> None:
        super().__init__()
        self.quantum_circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        squeezed = torch.squeeze(inputs) if inputs.dim() > 1 else inputs
        return HybridFunction.apply(squeezed, self.quantum_circuit, self.shift)

class HybridQuantumClassifier(nn.Module):
    """Hybrid quantum‑classical binary classifier mirroring the ML counterpart."""
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
        backend = qiskit.Aer.get_backend("aer_simulator")
        self.hybrid = Hybrid(n_qubits=2, backend=backend, shots=512, shift=np.pi / 2)

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
        x = self.hybrid(x).T
        probs = torch.sigmoid(x)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["FraudLayerParameters", "QuantumCircuit", "HybridFunction", "Hybrid", "HybridQuantumClassifier"]
