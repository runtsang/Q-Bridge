"""UnifiedClassifier: Quantum implementation using Qiskit.

The class builds a variational circuit that mirrors the classical
architecture.  It encodes input data with Rx rotations, applies a
depth‑controlled ansatz of Ry rotations and CZ entangling gates, and
returns the circuit together with metadata (encoding parameters,
variational parameters, observables) so that the interface matches the
classical version.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.providers.aer import AerSimulator


class UnifiedClassifier:
    """Quantum classifier with parameter‑ized encoding and ansatz."""

    def __init__(self, num_qubits: int, depth: int) -> None:
        """
        Parameters
        ----------
        num_qubits : int
            Number of qubits / input features.
        depth : int
            Number of variational layers.
        """
        self.num_qubits = num_qubits
        self.depth = depth

    def build_circuit(
        self,
    ) -> Tuple[QuantumCircuit, Iterable[ParameterVector], Iterable[ParameterVector], List[SparsePauliOp]]:
        """Build a layered circuit and return it with metadata."""
        # Encoding parameters
        encoding = ParameterVector("x", self.num_qubits)
        # Variational parameters
        weights = ParameterVector("theta", self.num_qubits * self.depth)

        circuit = QuantumCircuit(self.num_qubits)

        # Data encoding: Rx with input angles
        for qubit in range(self.num_qubits):
            circuit.rx(encoding[qubit], qubit)

        # Variational ansatz
        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                circuit.ry(weights[idx], qubit)
                idx += 1
            # Entanglement
            for qubit in range(self.num_qubits - 1):
                circuit.cz(qubit, qubit + 1)

        # Observables: Z on each qubit
        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits - i - 1))
            for i in range(self.num_qubits)
        ]

        weight_sizes = [self.num_qubits] * self.depth

        return circuit, list(encoding), list(weights), observables

    def forward(self, circuit: QuantumCircuit, params: dict) -> QuantumCircuit:
        """Apply a parameter binding to the circuit."""
        return circuit.bind_parameters(params)

    def kernel_matrix(self, x: List[float], y: List[float]) -> float:
        """Simple kernel evaluation using the circuit as a feature map.
        The kernel is defined as the absolute value of the overlap between
        states prepared from x and y with opposite signs."""
        circ, enc, _, _ = self.build_circuit()
        backend = AerSimulator()

        # Prepare state for x
        param_dict_x = {enc[i]: x[i] for i in range(self.num_qubits)}
        circ_x = circ.bind_parameters(param_dict_x)
        result_x = backend.run(circ_x).result()
        state_x = result_x.get_statevector()

        # Prepare state for y with negative signs
        param_dict_y = {enc[i]: -y[i] for i in range(self.num_qubits)}
        circ_y = circ.bind_parameters(param_dict_y)
        result_y = backend.run(circ_y).result()
        state_y = result_y.get_statevector()

        # Overlap
        overlap = np.abs(np.vdot(state_x, state_y))
        return overlap


__all__ = ["UnifiedClassifier"]
