"""Quantum classifier circuit factory with data‑re‑uploading ansatz.

Extends the original simple ansatz by allowing multiple encoding layers
and configurable entanglement patterns, and returns a list of observables
matching the number of qubits.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List, Optional

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


class QuantumClassifierModel:
    """Factory for a data‑re‑uploading variational circuit.

    The API mirrors the classical helper:
    - ``build(num_qubits, depth, encoding='rx', entanglement='cnot',
    observables=None)`` returns
      ``(circuit, encoding_params, weight_params, observables)``.
    """

    @staticmethod
    def build(
        num_qubits: int,
        depth: int,
        encoding: str = "rx",
        entanglement: str = "cnot",
        observables: Optional[List[SparsePauliOp]] = None,
    ) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
        """
        Parameters
        ----------
        num_qubits : int
            Number of qubits / input dimension.
        depth : int
            Number of data‑re‑uploading layers.
        encoding : str, default ``"rx"``
            Gate used for data encoding: ``"rx"``, ``"ry"``, or ``"rz"``.
        entanglement : str, default ``"cnot"``
            Entanglement pattern: ``"cnot"`` (CX chain) or ``"cx"`` (full nearest‑neighbor).
        observables : List[SparsePauliOp] | None
            List of measurement operators. If ``None``, defaults to Z on each qubit.

        Returns
        -------
        circuit : QuantumCircuit
            Variational circuit.
        encoding_params : Iterable
            ParameterVector for data encoding.
        weight_params : Iterable
            ParameterVector for variational weights.
        observables : List[SparsePauliOp]
            Measurement operators.
        """
        # Parameter vectors
        enc_params = ParameterVector(f"x_{encoding}", num_qubits)
        weight_params = ParameterVector("theta", num_qubits * depth)

        circuit = QuantumCircuit(num_qubits)

        for layer in range(depth):
            # Data encoding
            for q in range(num_qubits):
                gate = getattr(circuit, encoding)
                gate(enc_params[q], q)

            # Variational rotation
            idx = layer * num_qubits
            for q in range(num_qubits):
                circuit.ry(weight_params[idx + q], q)

            # Entanglement
            if entanglement == "cnot":
                for q in range(num_qubits - 1):
                    circuit.cx(q, q + 1)
            elif entanglement == "cx":
                for q in range(num_qubits):
                    target = (q + 1) % num_qubits
                    circuit.cx(q, target)
            else:
                raise ValueError(f"Unsupported entanglement pattern: {entanglement}")

        # Observables
        if observables is None:
            observables = [
                SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
                for i in range(num_qubits)
            ]

        return circuit, list(enc_params), list(weight_params), observables


__all__ = ["QuantumClassifierModel"]
