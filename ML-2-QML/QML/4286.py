"""Quantum hybrid layer that forwards activations through a parameterised
circuit and offers a differentiable expectation head.

`UnifiedHybridLayer` wraps a simple two‑qubit circuit, uses a finite‑difference
gradient estimator for back‑propagation, and exposes a `forward` method that
returns the quantum expectation value.  The module can be used as a drop‑in
replacement for the classical head in a hybrid network.
"""

import numpy as np
import torch
from torch import nn
from typing import Iterable
import qiskit
from qiskit import assemble, transpile

__all__ = ["UnifiedHybridLayer", "QuantumCircuitWrapper", "HybridFunction"]

class QuantumCircuitWrapper:
    """Parameterised two‑qubit circuit used as a quantum expectation head."""

    def __init__(self, n_qubits: int, backend, shots: int) -> None:
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.Parameter("theta")
        self._circuit.h(range(n_qubits))
        self._circuit.barrier()
        self._circuit.ry(self.theta, range(n_qubits))
        self._circuit.measure_all()
        self.backend = backend
        self.shots = shots

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        # Expectation of the Y observable
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        probs = counts / self.shots
        return np.sum(states * probs)

class HybridFunction(torch.autograd.Function):
    """Differentiable bridge between PyTorch and the quantum circuit."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        expectation = circuit.run(inputs.tolist())
        return torch.tensor([expectation], dtype=torch.float32)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Finite‑difference gradient estimation
        inputs = ctx.saved_tensors[0] if ctx.saved_tensors else None
        if inputs is None:
            return None, None, None
        shift = ctx.shift
        grads = []
        for val in inputs.tolist():
            exp_plus = ctx.circuit.run([val + shift])
            exp_minus = ctx.circuit.run([val - shift])
            grads.append(exp_plus - exp_minus)
        grads = torch.tensor(grads, dtype=torch.float32)
        return grads * grad_output, None, None

class UnifiedHybridLayer(nn.Module):
    """Quantum hybrid layer that forwards activations through a parameterised circuit."""

    def __init__(self, n_qubits: int = 2, shots: int = 100, shift: float = np.pi / 2) -> None:
        super().__init__()
        backend = qiskit.Aer.get_backend("aer_simulator")
        self.quantum_circuit = QuantumCircuitWrapper(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Runs the quantum circuit on the input tensor and returns the expectation."""
        return HybridFunction.apply(inputs, self.quantum_circuit, self.shift)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Convenience wrapper matching the classical interface."""
        values = torch.as_tensor(list(thetas), dtype=torch.float32)
        expectation = self.forward(values)
        return expectation.detach().cpu().numpy()
