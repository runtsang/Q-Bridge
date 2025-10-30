"""Quantum self‑attention block and hybrid components.

This module provides a pure quantum implementation of the self‑attention block
and a differentiable hybrid head. It is intended to be imported by the
classical module via ``from.quantum_self_attention import QuantumSelfAttention,
Hybrid, HybridFunction``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd

import qiskit
from qiskit import QuantumCircuit, transpile, assemble

__all__ = ["QuantumSelfAttention", "HybridFunction", "Hybrid", "HybridSelfAttentionNet"]

class QuantumSelfAttention(nn.Module):
    """Variational quantum circuit implementing a self‑attention block.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (and thus the dimensionality of the input).
    backend : qiskit.providers.basebackend.BaseBackend, optional
        Backend to execute the circuit. Defaults to the Aer QASM simulator.
    shots : int, optional
        Number of shots per circuit execution.
    """
    def __init__(self, n_qubits: int, backend=None, shots: int = 100):
        super().__init__()
        self.n_qubits = n_qubits
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.entangle_params = nn.Parameter(torch.randn(n_qubits - 1))
        self.rotation_params = nn.Parameter(torch.randn(n_qubits, 3))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size = inputs.shape[0]
        results = []
        for i in range(batch_size):
            circuit = QuantumCircuit(self.n_qubits, name=f"qsa_{i}")
            for q in range(self.n_qubits):
                circuit.rx(self.rotation_params[q, 0] * inputs[i, q].item(), q)
                circuit.ry(self.rotation_params[q, 1] * inputs[i, q].item(), q)
                circuit.rz(self.rotation_params[q, 2] * inputs[i, q].item(), q)
            for q in range(self.n_qubits - 1):
                circuit.crx(self.entangle_params[q].item(), q, q + 1)
            circuit.measure_all()
            compiled = transpile(circuit, self.backend)
            qobj = assemble(compiled, shots=self.shots)
            job = self.backend.run(qobj)
            result = job.result()
            counts = result.get_counts(circuit)
            expectations = []
            for q in range(self.n_qubits):
                zero = sum(counts.get(bin(1 << q)[2:].zfill(self.n_qubits), 0))
                one = self.shots - zero
                exp_z = (zero - one) / self.shots
                expectations.append(exp_z)
            results.append(expectations)
        return torch.tensor(results, dtype=torch.float32)

class HybridFunction(autograd.Function):
    """Differentiable interface that forwards activations through a quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumSelfAttention, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        expectation = ctx.circuit(inputs)
        result = torch.tensor(expectation, dtype=torch.float32)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = np.ones_like(inputs) * ctx.shift
        gradients = []
        for idx in range(inputs.shape[0]):
            right = ctx.circuit(inputs[idx] + shift[idx]).item()
            left = ctx.circuit(inputs[idx] - shift[idx]).item()
            gradients.append(right - left)
        gradients = torch.tensor(gradients, dtype=torch.float32)
        return gradients * grad_output, None, None

class Hybrid(nn.Module):
    """Hybrid layer that forwards a classical vector through a quantum circuit."""
    def __init__(self, n_qubits: int, backend=None, shots: int = 100, shift: float = np.pi / 2):
        super().__init__()
        self.quantum_circuit = QuantumSelfAttention(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs, self.quantum_circuit, self.shift)

class HybridSelfAttentionNet(nn.Module):
    """Quantum self‑attention block exposed as a PyTorch module."""
    def __init__(self, n_qubits: int, backend=None, shots: int = 100):
        super().__init__()
        self.quantum_sa = QuantumSelfAttention(n_qubits, backend, shots)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.quantum_sa(inputs)
