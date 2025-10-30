"""Unified hybrid layer with a differentiable quantum expectation head.

This version is tailored for quantum‑centric workflows.  It implements
a PyTorch autograd function that calls a Qiskit circuit and uses
finite‑difference gradients.  The class can be trained end‑to‑end on
CPU while the quantum part runs on a Qiskit Aer simulator or any
supported backend.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from collections.abc import Sequence
from typing import List

import qiskit
from qiskit import assemble, transpile


class _QuantumCircuitWrapper:
    """Parameterised two‑qubit (or arbitrary) Qiskit circuit."""
    def __init__(self, n_qubits: int = 1, backend=None, shots: int = 100):
        if backend is None:
            backend = qiskit.Aer.get_backend("qasm_simulator")
        self.backend = backend
        self.shots = shots
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        self._theta = qiskit.circuit.Parameter("theta")
        all_qubits = list(range(n_qubits))
        self._circuit.h(all_qubits)
        self._circuit.barrier()
        self._circuit.ry(self._theta, all_qubits)
        self._circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """Execute the circuit for a 1‑D array of angles and return an
        expectation value as a 1‑D numpy array."""
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self._theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts(self._circuit)

        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probabilities = counts / self.shots
            return np.sum(states * probabilities)

        if isinstance(result, list):
            return np.array([expectation(item) for item in result])
        return np.array([expectation(result)])

class HybridFunction(torch.autograd.Function):
    """Differentiable interface between PyTorch and the quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: _QuantumCircuitWrapper, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        ctx.save_for_backward(inputs)
        expectation = circuit.run(inputs.detach().cpu().numpy())
        return torch.tensor(expectation, dtype=torch.float32, device=inputs.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        shift = ctx.shift
        circuit = ctx.circuit
        inputs, = ctx.saved_tensors
        eps = 1e-3
        grad = []
        for val in inputs.detach().cpu().numpy():
            plus = circuit.run(np.array([val + eps]))
            minus = circuit.run(np.array([val - eps]))
            grad.append((plus - minus) / (2 * eps))
        grad_tensor = torch.tensor(grad, dtype=torch.float32, device=grad_output.device)
        return grad_tensor * grad_output, None, None

class Hybrid(nn.Module):
    """Hybrid layer wrapping a quantum circuit."""
    def __init__(self, n_qubits: int = 1, backend=None, shots: int = 100, shift: float = np.pi / 2):
        super().__init__()
        self.circuit = _QuantumCircuitWrapper(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs, self.circuit, self.shift)

class UnifiedHybridLayer(nn.Module):
    """Hybrid dense‑to‑quantum layer with batched evaluation.

    Parameters
    ----------
    n_features : int
        Number of input features for the classical dense block.
    n_qubits : int
        Number of qubits in the quantum circuit.
    backend : qiskit.providers.backend.Backend, optional
        Backend to run the quantum circuit.
    shots : int
        Number of shots for the quantum simulation.
    shift : float
        Finite‑difference shift used in the autograd function.
    """
    def __init__(
        self,
        n_features: int = 1,
        n_qubits: int = 1,
        backend=None,
        shots: int = 100,
        shift: float = np.pi / 2,
    ) -> None:
        super().__init__()
        self.dense = nn.Linear(n_features, 1)
        self.hybrid = Hybrid(n_qubits, backend, shots, shift)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the dense head followed by the quantum hybrid layer."""
        dense_out = torch.tanh(self.dense(inputs)).squeeze(-1)
        return self.hybrid(dense_out)

    def evaluate_batch(self, param_sets: Sequence[Sequence[float]]) -> List[float]:
        """Convenience wrapper to compute expectations for a list of inputs."""
        arr = np.array(param_sets)
        return list(self.hybrid.circuit.run(arr))

__all__ = ["UnifiedHybridLayer"]
