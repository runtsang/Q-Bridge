"""Unified hybrid quantum module for binary classification.

This module builds on the QCNet architecture but replaces the single‑qubit
head with a multi‑qubit variational layer that can be trained via
parameter‑shift gradients.  The quantum circuit is wrapped in a
PyTorch autograd Function so that the entire network remains differentiable.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import transpile, assemble
import torch
import torch.nn as nn
import torch.nn.functional as F

class _QuantumCircuit:
    """Reusable parametric circuit used by the hybrid layer."""
    def __init__(self, n_qubits: int, backend, shots: int, shift: float):
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.shift = shift

        # Simple variational ansatz
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.Parameter("theta")
        self.circuit.h(range(n_qubits))
        self.circuit.barrier()
        self.circuit.ry(self.theta, range(n_qubits))
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        if thetas.ndim == 1:
            thetas = thetas.reshape(-1, 1)
        exp_vals = []
        for theta in thetas:
            job = qiskit.execute(
                self.circuit,
                self.backend,
                shots=self.shots,
                parameter_binds=[{self.theta: theta[0]}],
            )
            counts = job.result().get_counts(self.circuit)
            probs = np.array(list(counts.values())) / self.shots
            states = np.array(list(counts.keys()), dtype=float)
            exp_vals.append(np.sum(states * probs))
        return np.array(exp_vals)

class HybridFunction(torch.autograd.Function):
    """Differentiable bridge between PyTorch and a quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: _QuantumCircuit):
        ctx.circuit = circuit
        # Run the circuit on the CPU for each input
        thetas = inputs.detach().cpu().numpy()
        exp_vals = circuit.run(thetas)
        out = torch.tensor(exp_vals, dtype=torch.float32).unsqueeze(-1)
        ctx.save_for_backward(inputs, out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, out = ctx.saved_tensors
        shift = np.ones_like(inputs) * ctx.circuit.shift
        grads = []
        for val in inputs.cpu().numpy():
            # Parameter‑shift rule
            exp_plus = ctx.circuit.run(np.array([val + shift[0]]))
            exp_minus = ctx.circuit.run(np.array([val - shift[0]]))
            grads.append(exp_plus - exp_minus)
        grads = torch.tensor(grads, dtype=torch.float32).unsqueeze(-1)
        return grads * grad_output, None

class Hybrid(nn.Module):
    """Hybrid head that forwards activations through a quantum circuit."""
    def __init__(self, n_qubits: int, backend, shots: int, shift: float):
        super().__init__()
        self.quantum = _QuantumCircuit(n_qubits, backend, shots, shift)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs, self.quantum)

class QCNet(nn.Module):
    """Convolutional network followed by a multi‑qubit quantum expectation head."""
    def __init__(self, n_qubits: int = 2, shift: float = np.pi / 2):
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
        self.hybrid = Hybrid(n_qubits, backend, shots=100, shift=shift)

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
        return torch.cat((x, 1 - x), dim=-1)

__all__ = ["HybridFunction", "Hybrid", "QCNet"]
