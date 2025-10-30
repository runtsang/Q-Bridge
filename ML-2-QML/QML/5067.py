"""Quantum‑aware components used by the hybrid network.

The module defines a circuit wrapper, an autograd function that
communicates with a quantum simulator, and a PyTorch layer that
exposes the quantum expectation as a differentiable head.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import assemble, transpile
from qiskit.circuit import Parameter
import torch
import torch.nn as nn

# --------------------------------------------------------------------------- #
# 1. Quantum circuit wrapper
# --------------------------------------------------------------------------- #
class QuantumCircuit:
    """Parameterized circuit that returns the expectation value of
    the Pauli‑Y operator for a given set of rotation angles.
    """

    def __init__(self, n_qubits: int, backend, shots: int) -> None:
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = Parameter("theta")
        for q in range(n_qubits):
            self._circuit.h(q)
            self._circuit.rx(self.theta, q)
        self._circuit.measure_all()
        self.backend = backend
        self.shots = shots

    def run(self, thetas: np.ndarray | list[list[float]]) -> np.ndarray:
        """Execute the circuit for a list of theta vectors.

        Returns an array of expectation values, one per input vector.
        """
        if isinstance(thetas, np.ndarray):
            thetas = thetas.tolist()
        elif not isinstance(thetas, list):
            raise TypeError("thetas must be a list or ndarray")

        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()

        def expectation(count_dict: dict[str, int]) -> float:
            counts = np.array(list(count_dict.values()))
            states = np.array([int(k, 2) for k in count_dict.keys()]).astype(float)
            probs = counts / self.shots
            return float(np.sum(states * probs))

        if isinstance(result, list):
            return np.array([expectation(r) for r in result])
        return np.array([expectation(result)])

# --------------------------------------------------------------------------- #
# 2. Autograd bridge
# --------------------------------------------------------------------------- #
class HybridFunction(torch.autograd.Function):
    """Bridge between PyTorch and the quantum circuit using the parameter‑shift rule."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.inputs = inputs
        ctx.circuit = circuit
        ctx.shift = shift
        thetas = inputs.float().tolist()
        expectation = circuit.run([thetas])[0]
        return torch.tensor([expectation], dtype=torch.float32)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs = ctx.inputs
        shift = ctx.shift
        circuit = ctx.circuit

        grad_inputs = []
        for i, theta in enumerate(inputs.tolist()):
            theta_plus = inputs.tolist()
            theta_minus = inputs.tolist()
            theta_plus[i] += shift
            theta_minus[i] -= shift
            exp_plus = circuit.run([theta_plus])[0]
            exp_minus = circuit.run([theta_minus])[0]
            grad_inputs.append(exp_plus - exp_minus)
        grad_tensor = torch.tensor(grad_inputs, dtype=torch.float32)
        return grad_tensor * grad_output[0], None, None

# --------------------------------------------------------------------------- #
# 3. Hybrid layer
# --------------------------------------------------------------------------- #
class Hybrid(nn.Module):
    """Layer that forwards a vector of angles through a quantum circuit."""

    def __init__(self, n_qubits: int, backend, shots: int, shift: float) -> None:
        super().__init__()
        self.circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim > 1:
            inputs = torch.squeeze(inputs)
        return HybridFunction.apply(inputs, self.circuit, self.shift)

__all__ = ["QuantumCircuit", "HybridFunction", "Hybrid"]
