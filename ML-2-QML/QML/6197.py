"""Hybrid fast estimator that integrates a classical model with a quantum circuit.

This implementation uses a Qiskit state‑vector simulator to obtain the
wavefunction for each set of parameters.  The hybrid layer forwards the
statevector to PyTorch, and the estimator evaluates arbitrary Qiskit
operators by explicit matrix‑vector multiplication.  The public API
matches the classical estimator but the underlying execution is quantum.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List

import numpy as np
import torch
from torch import nn
import qiskit
from qiskit import assemble, transpile
from qiskit.quantum_info import BaseOperator

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Return a 2‑D tensor from a sequence of floats."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class QuantumCircuitWrapper:
    """Parametrised circuit executed on a state‑vector backend."""

    def __init__(self, n_qubits: int, backend, shots: int):
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.param = qiskit.circuit.Parameter("θ")

        # Simple entangling pattern
        self.circuit.h(range(n_qubits))
        self.circuit.cx(0, 1)
        self.circuit.ry(self.param, 0)

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """Return the state‑vector for each theta in thetas."""
        states: List[np.ndarray] = []
        for theta in thetas:
            bound = self.circuit.assign_parameters({self.param: theta})
            compiled = transpile(bound, self.backend)
            qobj = assemble(compiled, shots=self.shots)
            job = self.backend.run(qobj)
            result = job.result()
            state = result.get_statevector(compiled)
            states.append(state)
        return np.array(states)

class HybridFunction(torch.autograd.Function):
    """Differentiable interface between PyTorch and the quantum circuit."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float):
        ctx.shift = shift
        ctx.circuit = circuit

        states = circuit.run(inputs.tolist())
        # Convert to torch tensor; shape (batch, 2**n_qubits)
        result = torch.tensor(states, dtype=torch.complex64)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        grads = []

        for val in inputs.squeeze().tolist():
            right = ctx.circuit.run([val + shift])
            left = ctx.circuit.run([val - shift])
            grads.append(right[0] - left[0])

        grad = torch.tensor(grads, dtype=torch.complex64)
        return grad.unsqueeze(0) * grad_output, None, None

class Hybrid(nn.Module):
    """Hybrid layer that forwards activations through a quantum circuit."""

    def __init__(self, n_qubits: int, backend, shots: int, shift: float):
        super().__init__()
        self.circuit = QuantumCircuitWrapper(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        flat = inputs.view(-1)
        return HybridFunction.apply(flat, self.circuit, self.shift)

class HybridFastEstimator:
    """Fast evaluation of a classical model followed by a quantum circuit."""

    def __init__(
        self,
        model: nn.Module,
        n_qubits: int,
        backend,
        shots: int = 100,
        shift: float = np.pi / 2,
    ) -> None:
        self.model = model
        self.quantum_layer = Hybrid(n_qubits, backend, shots, shift)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Evaluate observables for each parameter set.

        Parameters
        ----------
        observables
            Qiskit operators whose expectation values are computed.
        parameter_sets
            Iterable of parameter vectors to feed to the model.
        shots
            If provided, override the internal circuit shots.
        seed
            Unused but kept for API compatibility.
        """
        if shots is not None:
            self.quantum_layer.circuit.shots = shots

        observables = list(observables) or []
        results: List[List[complex]] = []

        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                raw = self.model(inputs)
                state = self.quantum_layer(raw)

                row = []
                for obs in observables:
                    mat = obs.to_matrix()
                    # state shape (batch, 2**n_qubits); take first sample
                    psi = state[0].cpu().numpy()
                    exp_val = psi.conj().T @ mat @ psi
                    row.append(exp_val)
                results.append(row)

        return results

__all__ = ["HybridFastEstimator"]
