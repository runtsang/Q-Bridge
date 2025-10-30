"""Quantum classifier circuit builder that extends the original
``build_classifier_circuit`` with an entangling ansatz and a
feature‑map encoding.  The interface mirrors the classical counterpart
so that the two modules can be swapped in experiments."""
from __future__ import annotations

from typing import Iterable, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


class QuantumClassifierModel:
    """Constructs a variational quantum circuit for binary classification.

    Parameters
    ----------
    num_qubits: int
        Number of qubits in the circuit.
    depth: int
        Number of variational layers.
    """

    def __init__(self, num_qubits: int, depth: int) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        (
            self.circuit,
            self.encoding_params,
            self.variational_params,
            self.observables,
        ) = self._build_circuit()

    def _build_circuit(
        self,
    ) -> Tuple[QuantumCircuit, Iterable, Iterable, list[SparsePauliOp]]:
        # Parameter vectors
        encoding = ParameterVector("x", self.num_qubits)
        variational = ParameterVector("theta", self.num_qubits * self.depth)

        qc = QuantumCircuit(self.num_qubits)

        # Feature‑map encoding: RX rotations
        for i, param in enumerate(encoding):
            qc.rx(param, i)

        # Variational ansatz
        idx = 0
        for _ in range(self.depth):
            # Single‑qubit rotations
            for q in range(self.num_qubits):
                qc.ry(variational[idx], q)
                idx += 1
            # Entangling layer: CZ chain
            for q in range(self.num_qubits - 1):
                qc.cz(q, q + 1)
            # Optional CNOT chain for added expressivity
            for q in range(self.num_qubits - 1):
                qc.cx(q, q + 1)

        # Observables: single‑qubit Z on each qubit
        obs = [
            SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits - i - 1))
            for i in range(self.num_qubits)
        ]

        return qc, list(encoding), list(variational), obs

    def get_parameter_count(self) -> int:
        """Return the number of variational parameters."""
        return len(self.variational_params)

    def __repr__(self) -> str:
        return (
            f"<QuantumClassifierModel num_qubits={self.num_qubits} "
            f"depth={self.depth}>"
        )


__all__ = ["QuantumClassifierModel"]
