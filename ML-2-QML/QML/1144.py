"""Quantum classifier with data‑re‑uploading and entangling layers.

The interface mirrors the classical helper: ``build_classifier_circuit`` returns
the circuit, encoding parameters, variational parameters, and a list of
observables (PauliZ on each qubit).  The ansatz uses repeated data encoding
followed by a block of single‑qubit rotations and a long‑range entangling
layer (CZ).  The depth controls the number of such blocks.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

class QuantumClassifierModel:
    """Factory for a data‑re‑uploading variational circuit.

    The circuit is built as follows:

    1. Data encoding: an RX rotation on each qubit with a dedicated
       parameter per feature.
    2. For ``depth`` iterations:
       a. Single‑qubit rotations (RY) with a unique parameter per qubit.
       b. A long‑range CZ entangling layer that couples each qubit to the
          next one (periodic boundary conditions).
       c. A second data encoding layer (re‑uploading) to increase expressivity.
    3. Measurements: Pauli‑Z on each qubit.

    The returned observables are ``SparsePauliOp`` objects that can be
    evaluated on the statevector or expectation‑value simulators.
    """

    @staticmethod
    def build_classifier_circuit(num_qubits: int,
                                 depth: int
                                 ) -> Tuple[QuantumCircuit,
                                            Iterable,
                                            Iterable,
                                            List[SparsePauliOp]]:
        """Construct the variational circuit.

        Parameters
        ----------
        num_qubits: int
            Number of qubits (also the number of input features).
        depth: int
            Number of variational blocks.

        Returns
        -------
        circuit: QuantumCircuit
            ``QuantumCircuit`` instance ready for simulation or execution.
        encoding: list[Parameter]
            ``ParameterVector`` for data encoding (``x_0,..., x_{n-1}``).
        weights: list[Parameter]
            ``ParameterVector`` for variational parameters (``theta_0,...``).
        observables: list[SparsePauliOp]
            List of ``SparsePauliOp`` objects measuring ``Z`` on each qubit.
        """
        encoding = ParameterVector("x", num_qubits)
        weights = ParameterVector("theta", num_qubits * depth)

        circuit = QuantumCircuit(num_qubits)

        # Initial data encoding
        for q, param in enumerate(encoding):
            circuit.rx(param, q)

        # Variational blocks with re‑uploading
        weight_index = 0
        for _ in range(depth):
            # Single‑qubit rotations
            for q in range(num_qubits):
                circuit.ry(weights[weight_index], q)
                weight_index += 1

            # Long‑range entangling CZ
            for q in range(num_qubits):
                circuit.cz(q, (q + 1) % num_qubits)

            # Re‑upload data
            for q, param in enumerate(encoding):
                circuit.rx(param, q)

        # Observables: Pauli‑Z on each qubit
        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
            for i in range(num_qubits)
        ]

        return circuit, list(encoding), list(weights), observables

__all__ = ["QuantumClassifierModel"]
