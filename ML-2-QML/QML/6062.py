"""QuantumClassifierModel: Qiskit quantum circuit with variational ansatz."""

from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import RealAmplitudes


class QuantumClassifierModel:
    """
    Quantum circuit factory that mimics the classical helper interface.

    The circuit consists of a data‑encoding layer followed by a stack of
    *RealAmplitudes* ansatz blocks.  Each block is entangled with a full
    CZ‑grid to introduce non‑linear correlations.  The observable set
    contains single‑qubit Z operators and an optional global ZZ‑observable
    for richer feature extraction.
    """

    def __init__(self, num_qubits: int, depth: int) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.circuit, self.encoding, self.weights, self.observables = self._build_circuit()

    def _build_circuit(self) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
        encoding = ParameterVector("x", self.num_qubits)
        ansatz = RealAmplitudes(num_qubits=self.num_qubits, reps=self.depth, entanglement="full")
        ansatz.params.name = "theta"

        circuit = QuantumCircuit(self.num_qubits)
        for idx, qubit in enumerate(range(self.num_qubits)):
            circuit.rx(encoding[idx], qubit)

        circuit.append(ansatz, range(self.num_qubits))

        # measurement observables
        observables = [SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits - i - 1)) for i in range(self.num_qubits)]
        observables.append(SparsePauliOp("Z" * self.num_qubits))  # global observable
        return circuit, list(encoding), list(ansatz.params), observables

    @staticmethod
    def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
        """
        Return the circuit and metadata analogous to the classical helper.
        """
        model = QuantumClassifierModel(num_qubits, depth)
        return model.circuit, model.encoding, model.weights, model.observables

    def __repr__(self) -> str:
        return f"<QuantumClassifierModel qubits={self.num_qubits} depth={self.depth}>"

__all__ = ["QuantumClassifierModel"]
