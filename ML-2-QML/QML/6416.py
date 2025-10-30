"""Quantum‑centric contribution for HybridQCNNNet.

Provides a differentiable quantum expectation head that can be
plugged into the classical network.  The head uses a single‑qubit
circuit with a parameterised rotation and the Aer simulator.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from qiskit import assemble, transpile
from qiskit.providers.aer import AerSimulator
import qiskit


class QuantumCircuitWrapper:
    """Simple 1‑qubit parameterised circuit."""

    def __init__(self, shots: int = 1024):
        self.backend = AerSimulator()
        self.shots = shots
        self.circuit = qiskit.QuantumCircuit(1)
        self.theta = qiskit.circuit.Parameter("theta")

        # Circuit definition
        self.circuit.h(0)
        self.circuit.ry(self.theta, 0)
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """Execute the circuit for the supplied angles."""
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()

        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys()), dtype=float)
            probs = counts / self.shots
            return np.sum(states * probs)

        return np.array([expectation(result)])


class QuantumHybridFunction(torch.autograd.Function):
    """Autograd wrapper that forwards to the quantum circuit."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float = 0.0) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        thetas = inputs.detach().cpu().numpy().squeeze()
        expectation = circuit.run(thetas)
        result = torch.tensor(expectation, dtype=inputs.dtype, device=inputs.device)
        ctx.save_for_backward(inputs)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, = ctx.saved_tensors
        shift = ctx.shift
        thetas = inputs.detach().cpu().numpy().squeeze()
        gradients = []
        for theta in thetas:
            right = ctx.circuit.run([theta + shift])
            left = ctx.circuit.run([theta - shift])
            gradients.append(right - left)
        grad = torch.tensor(gradients, dtype=inputs.dtype, device=inputs.device)
        return grad * grad_output, None, None


class QuantumHead(nn.Module):
    """Quantum expectation head used by :class:`HybridQCNNNet`."""

    def __init__(self, shift: float = np.pi / 2, shots: int = 1024):
        super().__init__()
        self.shift = shift
        self.circuit = QuantumCircuitWrapper(shots=shots)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return QuantumHybridFunction.apply(inputs, self.circuit, self.shift)


def compute_quantum_expectation(inputs: torch.Tensor) -> torch.Tensor:
    """
    Compute a probability from a quantum expectation value.

    Parameters
    ----------
    inputs : torch.Tensor
        Tensor of shape (batch, 1) containing the parameters for the
        quantum circuit.

    Returns
    -------
    torch.Tensor
        Tensor of shape (batch, 1) with values in [0, 1].
    """
    head = QuantumHead()
    out = head(inputs)
    # Map expectation [-1, 1] to [0, 1] using a sigmoid
    return torch.sigmoid(out)


__all__ = ["QuantumCircuitWrapper", "QuantumHybridFunction", "QuantumHead", "compute_quantum_expectation"]
