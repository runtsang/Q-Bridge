"""Quantum helper module for UnifiedEstimatorQNN.

Provides a parameterised two‑qubit circuit and a PyTorch autograd
function that evaluates its expectation value.  The circuit is
executed on a Qiskit Aer simulator and can be plugged into the
classical hybrid layer defined in the `ml` module.

The module mirrors the QuantumCircuit and HybridFunction from the
original ClassicalQuantumBinaryClassification example but is
rewritten to be lightweight and fully compatible with the new
UnifiedEstimatorQNN.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.circuit import Parameter
from qiskit.providers.aer import AerSimulator
from typing import Callable

# --------------------------------------------------------------------------- #
# Parameterised two‑qubit circuit
# --------------------------------------------------------------------------- #
class QuantumCircuitWrapper:
    """Two‑qubit parameterised circuit with a single observable."""
    def __init__(self, n_qubits: int = 2, backend: str = "aer_simulator", shots: int = 1024) -> None:
        self.n_qubits = n_qubits
        self.backend = AerSimulator()
        self.shots = shots

        self.circuit = QuantumCircuit(n_qubits)
        self.theta = Parameter("theta")
        self.circuit.h(range(n_qubits))
        self.circuit.ry(self.theta, range(n_qubits))
        self.circuit.measure_all()

    def run(self, params: np.ndarray) -> np.ndarray:
        """Evaluate the circuit for a batch of parameters."""
        params = np.asarray(params).flatten()
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(compiled,
                        shots=self.shots,
                        parameter_binds=[{self.theta: float(p)} for p in params])
        result = self.backend.run(qobj).result()
        counts = result.get_counts()
        # Expectation value of Z on qubit 0
        exp = 0.0
        for bitstring, cnt in counts.items():
            z = 1 if bitstring[-1] == "0" else -1
            exp += z * cnt / self.shots
        return np.array([exp])

# --------------------------------------------------------------------------- #
# Convenience wrapper for the quantum expectation
# --------------------------------------------------------------------------- #
def quantum_expectation_fn(params: np.ndarray) -> np.ndarray:
    """Convenience wrapper that can be passed to the hybrid layer."""
    qc = QuantumCircuitWrapper()
    return qc.run(params)

# --------------------------------------------------------------------------- #
# Autograd function that evaluates the quantum circuit
# --------------------------------------------------------------------------- #
class QuantumHybridFunction(torch.autograd.Function):
    """Autograd wrapper that evaluates a quantum circuit via the
    `quantum_expectation_fn` defined above.
    """
    @staticmethod
    def forward(ctx, inputs: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(inputs)
        # Convert to numpy
        params = inputs.detach().cpu().numpy()
        exp = quantum_expectation_fn(params)
        return torch.tensor(exp, dtype=inputs.dtype, device=inputs.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, = ctx.saved_tensors
        shift = np.pi / 2
        grad_inputs = []
        for i in range(inputs.shape[0]):
            plus = inputs.clone()
            minus = inputs.clone()
            plus[i] += shift
            minus[i] -= shift
            f_plus = quantum_expectation_fn(plus.detach().cpu().numpy())
            f_minus = quantum_expectation_fn(minus.detach().cpu().numpy())
            grad = (f_plus - f_minus) / (2 * shift)
            grad_inputs.append(grad)
        grad_inputs = torch.tensor(grad_inputs, dtype=inputs.dtype, device=inputs.device)
        return grad_inputs * grad_output

# --------------------------------------------------------------------------- #
# Hybrid layer that forwards through the quantum circuit
# --------------------------------------------------------------------------- #
class QuantumHybridLayer(nn.Module):
    """Hybrid layer that forwards through the quantum circuit."""
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return QuantumHybridFunction.apply(x)

__all__ = ["QuantumCircuitWrapper", "quantum_expectation_fn", "QuantumHybridFunction", "QuantumHybridLayer"]
