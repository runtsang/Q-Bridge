"""Quantum‑enhanced binary classifier with a quantum convolution filter.

This module implements the same architecture as the classical
HybridQCNet but replaces the final head with a parameterised
quantum circuit and optionally a quantum quanvolution filter.
The filter is implemented as a circuit that maps each image patch
to a probability of measuring |1> and is applied patch‑wise
to the input image.

The interface mirrors the classical version, so the two models
can be swapped without changing downstream code.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import assemble, transpile, execute
from qiskit.circuit.random import random_circuit
from qiskit.providers.aer import AerSimulator

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumCircuit:
    """Parameterized two‑qubit circuit for the hybrid head."""
    def __init__(self, n_qubits: int, backend: qiskit.providers.basebackend.BaseBackend, shots: int) -> None:
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
        """Execute the circuit for a batch of parameter values."""
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(compiled, shots=self.shots,
                        parameter_binds=[{self.theta: theta} for theta in thetas])
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array([int(k, 2) for k in count_dict.keys()])
            probs = counts / self.shots
            return np.sum(states * probs)
        if isinstance(result, list):
            return np.array([expectation(r) for r in result])
        return np.array([expectation(result)])


class HybridFunction(torch.autograd.Function):
    """Differentiable interface to the quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.quantum_circuit = circuit
        expectation_z = ctx.quantum_circuit.run(inputs.tolist())
        result = torch.tensor([expectation_z])
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = np.ones_like(inputs.tolist()) * ctx.shift
        grads = []
        for val in inputs.tolist():
            right = ctx.quantum_circuit.run([val + shift[0]])
            left = ctx.quantum_circuit.run([val - shift[0]])
            grads.append(right - left)
        grads = torch.tensor(grads).float()
        return grads * grad_output.float(), None, None


class Hybrid(nn.Module):
    """Hybrid head that forwards activations through a quantum circuit."""
    def __init__(self, n_qubits: int, backend: qiskit.providers.basebackend.BaseBackend,
                 shots: int, shift: float) -> None:
        super().__init__()
        self.quantum_circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs.squeeze(), self.quantum_circuit, self.shift)


class QuantumConvFilter:
    """Quantum quanvolution filter that maps a kernel‑sized patch to a probability."""
    def __init__(self, kernel_size: int = 2, backend: qiskit.providers.basebackend.BaseBackend = None,
                 shots: int = 100, threshold: float = 127) -> None:
        self.n_qubits = kernel_size ** 2
        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()
        self.backend = backend or AerSimulator()
        self.shots = shots
        self.threshold = threshold

    def run(self, data: np.ndarray) -> float:
        """Run the filter on a 2‑D array of shape (kernel_size, kernel_size)."""
        flat = data.reshape(1, self.n_qubits)
        binds = []
        for row in flat:
            bind = {}
            for i, val in enumerate(row):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            binds.append(bind)
        job = execute(self._circuit, self.backend, shots=self.shots, parameter_binds=binds)
        result = job.result().get_counts(self._circuit)
        counts = 0
        for key, val in result.items():
            ones = sum(int(b) for b in key)
            counts += ones * val
        return counts / (self.shots * self.n_qubits)


class HybridQCNet(nn.Module):
    """CNN followed by a quantum expectation head.

    The network optionally uses a quantum quanvolution filter
    before the first convolutional layer.  The final classification
    head is a variational circuit executed on Aer.
    """
    def __init__(self, use_filter: bool = False, filter_kwargs: dict | None = None) -> None:
        super().__init__()
        self.use_filter = use_filter
        if use_filter:
            self.filter = QuantumConvFilter(**(filter_kwargs or {}))
        else:
            self.filter = None

        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        backend = AerSimulator()
        self.hybrid = Hybrid(self.fc3.out_features, backend, shots=100, shift=np.pi / 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.use_filter:
            # Apply quantum filter patch‑wise (approximate)
            gray = inputs.mean(dim=1, keepdim=True)
            # Extract 2×2 patches
            patches = F.unfold(gray, kernel_size=2, stride=2)
            responses = []
            for i in range(patches.shape[1]):
                patch = patches[:, i].view(-1, 2, 2).cpu().numpy()
                responses.append(self.filter.run(patch))
            filter_map = torch.tensor(responses, device=inputs.device).unsqueeze(1)
            inputs = filter_map
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


__all__ = ["HybridQCNet", "QuantumConvFilter", "HybridFunction", "Hybrid"]
