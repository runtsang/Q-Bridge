"""HybridQCNet – quantum‑enhanced implementation using a two‑qubit expectation head.

The quantum head is a differentiable wrapper around a parameterised two‑qubit
circuit executed on a Qiskit Aer simulator.  The rest of the network
mirrors the classical version, enabling direct comparison between
classical and quantum layers.

Author: gpt-oss-20b
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import qiskit
from qiskit import assemble, transpile
from qiskit.circuit.random import random_circuit


class QuantumCircuit:
    """Two‑qubit circuit with a single rotation parameter."""

    def __init__(self, backend: qiskit.providers.BaseBackend, shots: int) -> None:
        self._circuit = qiskit.QuantumCircuit(2)
        self.theta = qiskit.circuit.Parameter("theta")

        # Simple entangling circuit
        self._circuit.h(0)
        self._circuit.h(1)
        self._circuit.barrier()
        self._circuit.ry(self.theta, 0)
        self._circuit.ry(self.theta, 1)
        self._circuit.measure_all()

        self.backend = backend
        self.shots = shots

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """Execute the circuit for each parameter value in `thetas`."""
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
        # Run the circuit for each input value
        expectations = circuit.run(inputs.detach().cpu().numpy())
        result = torch.tensor(expectations, dtype=torch.float32, device=inputs.device)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = np.full_like(inputs.detach().cpu().numpy(), ctx.shift)
        grads = []

        for value in inputs.detach().cpu().numpy():
            right = ctx.circuit.run([value + shift])
            left = ctx.circuit.run([value - shift])
            grads.append(right - left)

        grads = torch.tensor(grads, dtype=torch.float32, device=inputs.device)
        return grads * grad_output, None, None


class Hybrid(nn.Module):
    """Quantum expectation head."""

    def __init__(self, backend: qiskit.providers.BaseBackend, shots: int, shift: float = np.pi / 2) -> None:
        super().__init__()
        self.circuit = QuantumCircuit(backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs.squeeze(-1), self.circuit, self.shift)


class QuanvCircuit:
    """Quantum filter emulating a 2×2 convolution (quanvolution)."""

    def __init__(self, backend: qiskit.providers.BaseBackend, shots: int, threshold: float = 0.5) -> None:
        self.n_qubits = 4
        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.params = [qiskit.circuit.Parameter(f"θ{i}") for i in range(self.n_qubits)]

        for i in range(self.n_qubits):
            self._circuit.rx(self.params[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, depth=2)
        self._circuit.measure_all()

        self.backend = backend
        self.shots = shots
        self.threshold = threshold

    def run(self, data: np.ndarray) -> float:
        """Run the filter on a 2×2 patch and return the mean |1⟩ probability."""
        data = data.reshape(1, self.n_qubits)
        param_binds = []
        for row in data:
            bind = {}
            for idx, val in enumerate(row):
                bind[self.params[idx]] = np.pi if val > self.threshold else 0.0
            param_binds.append(bind)

        job = qiskit.execute(self._circuit, self.backend, shots=self.shots, parameter_binds=param_binds)
        result = job.result().get_counts(self._circuit)

        total = 0
        for key, val in result.items():
            ones = sum(int(b) for b in key)
            total += ones * val
        return total / (self.shots * self.n_qubits)


class HybridQCNet(nn.Module):
    """CNN + fully‑connected head with a quantum expectation output."""

    def __init__(self, backend: qiskit.providers.BaseBackend, shots: int = 200) -> None:
        super().__init__()
        # Classical convolutional backbone
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Fully‑connected layers
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Quantum hybrid head
        self.hybrid = Hybrid(backend, shots, shift=np.pi / 2)

        # Optional quanvolution filter (unused in the forward path but available for experiments)
        self.quanv = QuanvCircuit(backend, shots, threshold=0.5)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
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
        probs = self.hybrid(x).squeeze(-1)
        return torch.stack([probs, 1 - probs], dim=-1)


__all__ = ["QuantumCircuit", "HybridFunction", "Hybrid", "QuanvCircuit", "HybridQCNet"]
