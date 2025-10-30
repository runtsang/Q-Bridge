"""Quantum hybrid CNN for binary classification.

This module implements the same architecture as the classical
counterpart but replaces the final head with a parameterised
four‑qubit quantum circuit.  The circuit is executed on the Aer
simulator and the expectation values of Pauli‑Z are combined into a
binary probability.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import assemble, transpile

class QuantumCircuit:
    """Parameterised multi‑qubit circuit executed on Aer."""

    def __init__(self, n_qubits: int, backend, shots: int, shift: float) -> None:
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.shift = shift
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        all_qubits = list(range(n_qubits))
        self.theta = qiskit.circuit.Parameter("theta")
        # Prepare equal superposition
        self._circuit.h(all_qubits)
        # Parameterised Ry on each qubit
        for q in all_qubits:
            self._circuit.ry(self.theta, q)
        # Entanglement pattern: CX between consecutive qubits
        for q in range(n_qubits - 1):
            self._circuit.cx(q, q + 1)
        self._circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """Execute the circuit for a batch of parameters."""
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()

        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probabilities = counts / self.shots
            # Map state string to integer
            ints = np.array([int(s, 2) for s in states])
            # Compute expectation of Z for each qubit
            expectations = []
            for q in range(self.n_qubits):
                mask = 1 << (self.n_qubits - 1 - q)
                bits = (ints & mask) >> (self.n_qubits - 1 - q)
                exp = np.sum((1 - 2 * bits) * probabilities)
                expectations.append(exp)
            return np.array(expectations)

        if isinstance(result, list):
            return np.array([expectation(item) for item in result])
        return expectation(result)


class QuantumHybridFunction(torch.autograd.Function):
    """Differentiable interface between PyTorch and the quantum circuit."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.quantum_circuit = circuit
        expectations = ctx.quantum_circuit.run(inputs.tolist())
        result = torch.tensor(expectations, dtype=torch.float32, device=inputs.device)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = np.ones_like(inputs.tolist()) * ctx.shift
        grad_inputs = []
        for idx, val in enumerate(inputs.tolist()):
            right = ctx.quantum_circuit.run([val + shift[idx]])
            left = ctx.quantum_circuit.run([val - shift[idx]])
            grad = (right - left) / 2.0
            grad_inputs.append(grad)
        grad_inputs = torch.tensor(grad_inputs, dtype=torch.float32, device=inputs.device)
        return grad_inputs * grad_output, None, None


class QuantumHybrid(nn.Module):
    """Hybrid layer that forwards activations through a quantum circuit."""

    def __init__(self, n_qubits: int, backend, shots: int, shift: float) -> None:
        super().__init__()
        self.quantum_circuit = QuantumCircuit(n_qubits, backend, shots, shift)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return QuantumHybridFunction.apply(inputs, self.quantum_circuit, self.shift)


class HybridQuantumCNN(nn.Module):
    """Convolutional network followed by a 4‑qubit quantum expectation head."""

    def __init__(self, n_qubits: int = 4, shots: int = 100, shift: float = np.pi / 2) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_qubits)
        backend = qiskit.Aer.get_backend("aer_simulator")
        self.hybrid = QuantumHybrid(n_qubits, backend, shots=shots, shift=shift)

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
        expectations = self.hybrid(x).T
        # Reduce to a single probability by summing expectations and applying sigmoid
        prob = torch.sigmoid(expectations.sum(dim=1, keepdim=True))
        return torch.cat((prob, 1 - prob), dim=-1)


__all__ = ["QuantumCircuit", "QuantumHybridFunction", "QuantumHybrid", "HybridQuantumCNN"]
