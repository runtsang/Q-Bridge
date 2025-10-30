"""Quantum implementation of the hybrid QCNet with a fully connected quantum layer.

The module extends the original QCNet by replacing the linear head with a
parameterised quantum fully connected layer (`QuantumFCL`) that operates
on a two‑qubit Aer simulator.  The hybrid function uses a finite‑difference
gradient estimator compatible with PyTorch autograd, enabling end‑to‑end
optimisation.  The design follows the seed code but adds:
* `QuantumFCL` – a quantum analogue of the classical `FCL`.
* `HybridFunction` – accepts a quantum circuit and shift, returning a
  differentiable expectation value.
* `Hybrid` – a hybrid head that can toggle between a classical linear
  layer and the quantum fully connected layer.
* `QCNet` – can be instantiated with either head for comparative studies.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import qiskit
from qiskit import assemble, transpile


class QuantumCircuit:
    """Parameterized two‑qubit circuit executed on Aer."""
    def __init__(self, n_qubits: int, backend, shots: int):
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        all_qubits = list(range(n_qubits))
        self.theta = qiskit.circuit.Parameter("theta")
        self._circuit.h(all_qubits)
        self._circuit.barrier()
        self._circuit.ry(self.theta, all_qubits)
        self._circuit.measure_all()
        self.backend = backend
        self.shots = shots

    def run(self, thetas: np.ndarray) -> np.ndarray:
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
            return np.sum(states * probabilities)

        if isinstance(result, list):
            return np.array([expectation(item) for item in result])
        return np.array([expectation(result)])


class QuantumFCL:
    """Quantum fully connected layer mirroring the classical FCL."""
    def __init__(self, n_qubits: int, backend, shots: int):
        self.circuit = QuantumCircuit(n_qubits, backend, shots)

    def run(self, thetas: np.ndarray) -> np.ndarray:
        return self.circuit.run(thetas)


class HybridFunction(torch.autograd.Function):
    """Differentiable interface between PyTorch and a quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        expectation_z = ctx.circuit.run(inputs.tolist())
        result = torch.tensor(expectation_z, device=inputs.device)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        input_values = np.array(inputs.tolist())
        shift = np.ones_like(input_values) * ctx.shift
        gradients = []
        for idx, value in enumerate(input_values):
            expectation_right = ctx.circuit.run([value + shift[idx]])
            expectation_left = ctx.circuit.run([value - shift[idx]])
            gradients.append(expectation_right - expectation_left)
        gradients = torch.tensor(gradients, device=inputs.device, dtype=torch.float32)
        return gradients * grad_output, None, None


class Hybrid(nn.Module):
    """Hybrid head that can switch between a linear layer and a quantum FCL."""
    def __init__(self, n_features: int, use_quantum: bool = False,
                 n_qubits: int = 2, shots: int = 100, shift: float = np.pi / 2):
        super().__init__()
        self.use_quantum = use_quantum
        self.shift = shift
        if use_quantum:
            backend = qiskit.Aer.get_backend("aer_simulator")
            self.quantum_fcl = QuantumFCL(n_qubits, backend, shots)
        else:
            self.linear = nn.Linear(n_features, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.use_quantum:
            squeezed = torch.squeeze(inputs)
            return HybridFunction.apply(squeezed, self.quantum_fcl.circuit, self.shift)
        else:
            logits = self.linear(inputs)
            return torch.sigmoid(logits + self.shift)


class QCNet(nn.Module):
    """Convolutional network followed by a hybrid quantum or linear head."""
    def __init__(self, use_quantum: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.hybrid = Hybrid(1, use_quantum=use_quantum)

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
        x = self.fc3(x).view(-1, 1)
        logits = self.hybrid(x)
        probs = torch.cat((logits, 1 - logits), dim=-1)
        return probs
