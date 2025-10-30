"""Quantum circuit factory for the QuantumClassifierModel project.

This module introduces an enriched ansatz that supports:
- Parameter‑efficient data encoding with Ry rotations.
- Configurable entanglement layers (CZ or CX) with optional skip
  connections.
- A flexible observable set, returning a list of ``SparsePauliOp``.
The :class:`QuantumClassifierModel` class mirrors the classical
counterpart but operates on Qiskit objects.

Typical usage::

    from qml_module import QuantumClassifierModel
    circ, enc, params, obs = QuantumClassifierModel.build_classifier_circuit(
        num_qubits=4, depth=2, entanglement='cz')

    backend = Aer.get_backend('qasm_simulator')
    job = execute(circ, backend, shots=1024)
    result = job.result()
    counts = result.get_counts()

"""

from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit import Aer, execute


class QuantumClassifierModel:
    """Shared quantum classifier class.

    The :meth:`build_classifier_circuit` factory returns a data‑encoding
    circuit together with parameter lists and measurement observables,
    ready for integration with hybrid training loops.
    """

    @staticmethod
    def build_classifier_circuit(
        num_qubits: int,
        depth: int,
        entanglement: str = "cz",
        encoding_gate: str = "ry",
    ) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
        """
        Construct a layered variational ansatz with explicit encoding.

        Parameters
        ----------
        num_qubits:
            Number of qubits in the circuit.
        depth:
            Number of variational layers.
        entanglement:
            Type of two‑qubit entanglement.  Options are ``'cz'`` or ``'cx'``.
        encoding_gate:
            Gate used for data encoding.  Currently ``'ry'`` or ``'rz'``.

        Returns
        -------
        circuit:
            Qiskit :class:`QuantumCircuit` instance.
        encoding:
            List of :class:`ParameterVector` objects for data encoding.
        weights:
            List of :class:`ParameterVector` objects for variational angles.
        observables:
            List of :class:`SparsePauliOp` measuring each qubit in the Z basis.
        """
        if encoding_gate not in {"ry", "rz"}:
            raise ValueError("encoding_gate must be 'ry' or 'rz'")
        if entanglement not in {"cz", "cx"}:
            raise ValueError("entanglement must be 'cz' or 'cx'")

        # Data‑encoding parameters
        encoding = ParameterVector(f"x", num_qubits)

        # Variational parameters: one angle per qubit per layer
        weights = ParameterVector(f"theta", num_qubits * depth)

        circ = QuantumCircuit(num_qubits)

        # Data encoding
        for idx, qubit in enumerate(range(num_qubits)):
            if encoding_gate == "ry":
                circ.ry(encoding[idx], qubit)
            else:
                circ.rz(encoding[idx], qubit)

        # Variational layers
        param_idx = 0
        for _ in range(depth):
            # Single‑qubit rotations
            for qubit in range(num_qubits):
                circ.rz(weights[param_idx], qubit)
                param_idx += 1

            # Entangling layer
            if entanglement == "cz":
                for qubit in range(num_qubits - 1):
                    circ.cz(qubit, qubit + 1)
            else:  # cx
                for qubit in range(num_qubits - 1):
                    circ.cx(qubit, qubit + 1)

            # Optional skip connection (identity) can be added here

        # Observables: measure each qubit in Z basis
        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
            for i in range(num_qubits)
        ]

        return circ, encoding, weights, observables


__all__ = ["QuantumClassifierModel"]
