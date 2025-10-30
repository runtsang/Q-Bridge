"""Quantum implementation of the hybrid classifier.

The class builds a parameterised circuit that mirrors the structure of the
classical model: a feature‑encoding layer, a depth‑scaled variational ansatz,
and simple Pauli‑Z observables.  It returns a SamplerQNN instance that can
be used as a differentiable layer in a quantum‑classical training loop.

Architecture
------------
* Feature encoding: RX rotations per qubit
* Variational ansatz: depth‑scaled layers of Ry rotations + CZ entanglement
* Observables: two Z‑like Pauli operators providing two outputs
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN


class HybridQuantumClassifier:
    """
    Quantum counterpart of :class:`HybridQuantumClassifier` from the classical
    module.  It constructs a circuit with an explicit feature‑encoding layer
    followed by a depth‑scaled variational ansatz.  The circuit is wrapped in
    a :class:`SamplerQNN` so that it can be differentiated and trained
    with a classical optimiser.
    """

    def __init__(self, num_qubits: int, depth: int = 2) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.circuit, self.encoding, self.weights, self.observables = self._build_circuit()

    def _build_circuit(self) -> tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
        """Create a variational circuit with explicit encoding and measurement."""
        encoding = ParameterVector("x", self.num_qubits)
        weights = ParameterVector("theta", self.num_qubits * self.depth)

        qc = QuantumCircuit(self.num_qubits)

        # Feature encoding: RX rotation per qubit
        for param, qubit in zip(encoding, range(self.num_qubits)):
            qc.rx(param, qubit)

        # Variational ansatz
        w_idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                qc.ry(weights[w_idx], qubit)
                w_idx += 1
            for qubit in range(self.num_qubits - 1):
                qc.cz(qubit, qubit + 1)

        # Simple measurement observables: two Z‑like Pauli operators
        observables = [
            SparsePauliOp("Z" + "I" * (self.num_qubits - 1)),
            SparsePauliOp("I" + "Z" + "I" * (self.num_qubits - 2)),
        ]

        return qc, list(encoding), list(weights), observables

    def get_qnn(self) -> SamplerQNN:
        """Return a SamplerQNN that can be used as a differentiable layer."""
        sampler = Sampler()
        return SamplerQNN(
            circuit=self.circuit,
            input_params=self.encoding,
            weight_params=self.weights,
            interpret=lambda x: x,
            output_shape=2,
            sampler=sampler,
        )


__all__ = ["HybridQuantumClassifier"]
