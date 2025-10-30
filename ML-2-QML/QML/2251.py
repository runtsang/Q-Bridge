"""Quantum estimator utilities for hybrid models.

The module defines a lightweight state‑vector simulator wrapper
and a differentiable hybrid layer that can be dropped into a
PyTorch network.  The design follows the original seed but adds
parameter‑shift gradients and a simple expectation‑value interface.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Iterable, List

from qiskit import QuantumCircuit as QC
from qiskit.quantum_info import Statevector, Z
from qiskit.providers.basicaer import AerSimulator

# --------------------------------------------------------------------------- #
#  Quantum circuit wrapper
# --------------------------------------------------------------------------- #
class QuantumCircuit:
    """Parameterized two‑qubit circuit that returns the Z expectation.

    The circuit is a simple H‑RY‑measure construction that is
    cheap to simulate with the Aer simulator.  It can be used
    either deterministically (state‑vector) or with a finite
    number of shots.
    """
    def __init__(self, n_qubits: int = 2, shots: int | None = None) -> None:
        self._circuit = QC(n_qubits)
        self._circuit.h(range(n_qubits))
        self.theta = QC.Parameter("theta")
        self._circuit.ry(self.theta, range(n_qubits))
        self._circuit.measure_all()

        self.backend = AerSimulator()
        self.shots = shots

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """Execute the circuit for a batch of angles.

        Parameters
        ----------
        thetas:
            1‑D array of angles.  The length must match the number of
            parameters of the circuit (here 1, but the interface
            accepts a vector for future extensibility).
        Returns
        -------
        expectation:
            1‑D array of expectation values for each angle.
        """
        expectations = []
        for theta in thetas:
            bound = self._circuit.assign_parameters({self.theta: theta})
            state = Statevector.from_instruction(bound)
            expectation = state.expectation_value(Z)
            expectations.append(float(expectation))
        return np.array(expectations)

# --------------------------------------------------------------------------- #
#  Differentiable hybrid layer
# --------------------------------------------------------------------------- #
class HybridFunction(torch.autograd.Function):
    """Bridge between PyTorch and the quantum circuit.

    The forward pass evaluates the circuit expectation.  The backward
    pass uses the parameter‑shift rule to compute gradients.
    """
    @staticmethod
    def forward(
        ctx,
        inputs: torch.Tensor,
        circuit: QuantumCircuit,
        shift: float = np.pi / 2,
    ) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        angles = inputs.detach().cpu().numpy()
        expectation = circuit.run(angles)
        result = torch.tensor(expectation, dtype=torch.float32, device=inputs.device)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        angles = inputs.detach().cpu().numpy()
        grads = []
        for idx, angle in enumerate(angles):
            right = ctx.circuit.run([angle + shift])
            left = ctx.circuit.run([angle - shift])
            grads.append(right - left)
        grads = torch.tensor(grads, dtype=torch.float32, device=grad_output.device)
        return grads * grad_output, None, None

class HybridLayer(nn.Module):
    """Hybrid layer that forwards activations through a quantum circuit."""
    def __init__(
        self,
        n_qubits: int = 2,
        shots: int | None = None,
        shift: float = np.pi / 2,
    ) -> None:
        super().__init__()
        self.circuit = QuantumCircuit(n_qubits, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        flat = inputs.view(inputs.size(0), -1)
        return HybridFunction.apply(flat, self.circuit, self.shift)

__all__ = ["QuantumCircuit", "HybridFunction", "HybridLayer"]
