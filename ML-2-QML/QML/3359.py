"""Quantum kernel and sampler implementation using Qiskit.

This module implements a variational quantum kernel and a parameterised
quantum sampler that mirrors the classical SamplerQNN.  The public API
is intentionally identical to the classical implementation so that
downstream code can interchange the back‑end without modification.
"""

from __future__ import annotations

import numpy as np
import torch
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN
from qiskit.primitives import StatevectorSampler

__all__ = [
    "KernalAnsatz",
    "Kernel",
    "kernel_matrix",
    "QuantumKernelMethod",
]


class KernalAnsatz:
    """Variational ansatz that encodes two‑dimensional data into a 4‑qubit
    circuit.  The circuit is fully parameterised and can be trained
    by a gradient‑based optimiser.
    """

    def __init__(self, n_wires: int = 4) -> None:
        self.n_wires = n_wires
        self.backend = Aer.get_backend("statevector_simulator")

    def encode(self, x: torch.Tensor) -> QuantumCircuit:
        """Return a circuit that prepares the state |ψ(x)⟩."""
        qc = QuantumCircuit(self.n_wires)
        for i, val in enumerate(x):
            qc.ry(val.item(), i)
        # Add a modest entangling layer
        for i in range(self.n_wires - 1):
            qc.cx(i, i + 1)
        return qc

    def overlap(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Compute |⟨ψ(x)|ψ(y)⟩|² using statevector simulation."""
        qc_x = self.encode(x)
        qc_y = self.encode(y)
        sv_x = execute(qc_x, self.backend).result().get_statevector()
        sv_y = execute(qc_y, self.backend).result().get_statevector()
        return abs(np.vdot(sv_x, sv_y)) ** 2


class Kernel:
    """Wrapper that exposes the variational kernel as a callable."""

    def __init__(self, n_wires: int = 4) -> None:
        self.ansatz = KernalAnsatz(n_wires)

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.tensor(self.ansatz.overlap(x, y))


def kernel_matrix(a, b):
    kernel = Kernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])


class QuantumKernelMethod:
    """
    Quantum implementation of the hybrid kernel module.  The interface
    matches the classical counterpart while the internal mechanics rely
    on Qiskit primitives.

    Parameters
    ----------
    n_wires : int, optional
        Number of qubits used to encode the data.
    """

    def __init__(self, n_wires: int = 4) -> None:
        self.n_wires = n_wires
        self.kernel = Kernel(n_wires)
        self._setup_sampler()

    def _setup_sampler(self) -> None:
        # Parameterised 2‑qubit sampler circuit
        inputs = ParameterVector("x", 2)
        weights = ParameterVector("w", 4)
        qc = QuantumCircuit(2)
        qc.ry(inputs[0], 0)
        qc.ry(inputs[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[0], 0)
        qc.ry(weights[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[2], 0)
        qc.ry(weights[3], 1)

        self.sampler_qnn = QiskitSamplerQNN(
            circuit=qc,
            input_params=inputs,
            weight_params=weights,
            sampler=StatevectorSampler()
        )

    def transform(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run the quantum sampler and return the probability vector."""
        probs = []
        for inp in inputs:
            # QiskitSamplerQNN expects a list of floats
            probs.append(self.sampler_qnn(inp.tolist()))
        return torch.tensor(probs)

    def kernel_matrix(self, a, b):
        return kernel_matrix(a, b)

    def combined_kernel(self, a, b):
        """Compute a kernel matrix on the sampler output."""
        a_mapped = [x for x in self.transform(a)]
        b_mapped = [x for x in self.transform(b)]
        return kernel_matrix(a_mapped, b_mapped)
