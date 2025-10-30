"""Quantum module providing a variational circuit and a differentiable hybrid layer.

The quantum circuit is a simple parameterised Ry‑gate network that returns
a single expectation value.  A custom autograd function forwards a tensor
through this circuit and approximates gradients via finite differences.
The module is meant to be imported by the classical SelfAttentionHybrid
class defined in the ml_code above.

Dependencies
-------------
* torch
* numpy
* qiskit
"""

import numpy as np
import torch
import torch.nn as nn
import qiskit
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.circuit import Parameter


class QuantumAttentionCircuit:
    """Parametrised variational circuit returning one expectation value."""

    def __init__(self, n_qubits: int, backend, shots: int = 512):
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots

        theta = Parameter("θ")
        self.theta = theta

        self.circuit = QuantumCircuit(n_qubits)
        for q in range(n_qubits):
            self.circuit.h(q)
            self.circuit.ry(theta, q)
        self.circuit.measure_all()

    def run(self, params: np.ndarray) -> np.ndarray:
        """Execute the circuit for each parameter value in params."""
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: p} for p in params],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()

        def expectation(counts):
            probs = np.array(list(counts.values())) / self.shots
            states = np.array([int(s, 2) for s in counts.keys()])
            return np.sum(states * probs)

        if isinstance(result, list):
            return np.array([expectation(r) for r in result])
        return np.array([expectation(result)])


class HybridFunction(torch.autograd.Function):
    """Differentiable wrapper that forwards a tensor through the quantum circuit."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumAttentionCircuit, shift: float):
        ctx.shift = shift
        ctx.circuit = circuit

        # Aggregate inputs to a scalar per sample (mean)
        agg = inputs.mean(dim=-1) if inputs.dim() > 1 else inputs
        params = agg.detach().cpu().numpy()
        exp = circuit.run(params)
        out = torch.tensor(exp, device=inputs.device, dtype=torch.float32)
        ctx.save_for_backward(inputs, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        grad_inputs = torch.zeros_like(inputs)

        # Finite‑difference gradient for each element
        for i in range(inputs.shape[0]):
            for j in range(inputs.shape[1]):
                pert_plus = inputs.clone()
                pert_minus = inputs.clone()
                pert_plus[i, j] += shift
                pert_minus[i, j] -= shift
                exp_plus = ctx.circuit.run(pert_plus[i].mean().item())
                exp_minus = ctx.circuit.run(pert_minus[i].mean().item())
                grad = (exp_plus - exp_minus) / (2 * shift)
                grad_inputs[i, j] = grad
        return grad_inputs * grad_output, None, None


class Hybrid(nn.Module):
    """Hybrid layer that maps a feature vector to a scalar via a quantum circuit."""

    def __init__(self, n_qubits: int, backend, shots: int = 256, shift: float = np.pi / 2):
        super().__init__()
        self.circuit = QuantumAttentionCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs, self.circuit, self.shift)


__all__ = ["QuantumAttentionCircuit", "HybridFunction", "Hybrid"]
