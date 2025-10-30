"""Quantum hybrid classifier with a variational circuit and learnable shift."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import assemble, transpile
from qiskit.providers.aer import AerSimulator

class VariationalQuantumCircuit:
    """Parameterized two‑qubit variational circuit."""

    def __init__(self, backend: qiskit.providers.BaseBackend, shots: int = 1024) -> None:
        self.backend = backend
        self.shots = shots
        self.circuit = qiskit.QuantumCircuit(2)
        self.theta = qiskit.circuit.Parameter("theta")
        # Basic entangling circuit
        self.circuit.h(0)
        self.circuit.h(1)
        self.circuit.cx(0, 1)
        self.circuit.ry(self.theta, 0)
        self.circuit.ry(self.theta, 1)
        self.circuit.measure_all()

    def run(self, theta_values: np.ndarray) -> np.ndarray:
        """Run the circuit for each theta in theta_values."""
        expectations = []
        for val in theta_values:
            bind = {self.theta: val}
            bound_circ = self.circuit.bind_parameters(bind)
            compiled = transpile(bound_circ, self.backend)
            qobj = assemble(compiled, shots=self.shots)
            job = self.backend.run(qobj)
            result = job.result()
            counts = result.get_counts()
            probs = np.array(list(counts.values())) / self.shots
            states = np.array([int(k, 2) for k in counts.keys()])
            expectations.append(np.sum(states * probs))
        return np.array(expectations)

class HybridFunction(torch.autograd.Function):
    """Differentiable interface between PyTorch and the variational circuit."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: VariationalQuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        # Convert inputs to numpy array
        theta_vals = inputs.detach().cpu().numpy()
        expectations = circuit.run(theta_vals)
        result = torch.tensor(expectations, dtype=inputs.dtype, device=inputs.device)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        # Finite difference gradient
        eps = 1e-3
        grad_inputs = torch.zeros_like(inputs)
        for i in range(inputs.shape[0]):
            theta_plus = inputs[i] + eps
            theta_minus = inputs[i] - eps
            exp_plus = ctx.circuit.run(np.array([theta_plus.item()]))
            exp_minus = ctx.circuit.run(np.array([theta_minus.item()]))
            grad = (exp_plus - exp_minus) / (2 * eps)
            grad_inputs[i] = grad
        return grad_inputs * grad_output, None, None

class QuantumHybridClassifier(nn.Module):
    """Convolutional backbone followed by a variational quantum head."""

    def __init__(self, shift_init: float = 0.0, shots: int = 1024) -> None:
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        # Fully connected layers
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        # Quantum circuit
        backend = AerSimulator()
        self.quantum = VariationalQuantumCircuit(backend, shots=shots)
        self.shift = shift_init

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
        x = self.fc3(x).squeeze(-1)
        # Quantum head
        quantum_out = HybridFunction.apply(x, self.quantum, self.shift)
        probs = torch.sigmoid(quantum_out + self.shift)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["QuantumHybridClassifier"]
