from __future__ import annotations

import qiskit as qk
import qiskit.quantum_info as qinfo
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from typing import Iterable, Tuple, List

class UnifiedQuantumClassifier:
    """
    Quantum‑centric classifier that mirrors the classical API.  It
    constructs a variational circuit with data‑encoding, entangling
    layers, and measurement observables.  The circuit is inspired by
    the QTransformer and QuantumClassifier seeds.
    """
    def __init__(self, num_qubits: int, depth: int, num_classes: int = 2) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.num_classes = num_classes

    def build(self) -> QuantumCircuit:
        encoding = ParameterVector("x", self.num_qubits)
        weights = ParameterVector("theta", self.num_qubits * self.depth)

        qc = QuantumCircuit(self.num_qubits)

        # Data‑encoding layer
        for i, qubit in enumerate(range(self.num_qubits)):
            qc.rx(encoding[i], qubit)

        # Variational layers
        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                qc.ry(weights[idx], qubit)
                idx += 1
            # Entanglement
            for qubit in range(self.num_qubits - 1):
                qc.cz(qubit, qubit + 1)
            qc.cz(self.num_qubits - 1, 0)

        return qc

def build_classifier_circuit(
    num_qubits: int,
    depth: int,
    *,
    num_classes: int = 2,
) -> Tuple[QuantumCircuit, Iterable[ParameterVector], Iterable[ParameterVector], List[qinfo.SparsePauliOp]]:
    """
    Factory that returns a quantum circuit and associated metadata.
    """
    circuit = UnifiedQuantumClassifier(num_qubits, depth, num_classes).build()

    # Parameter vectors
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    # Observables: measure Z on each qubit
    observables = [qinfo.SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]

    return circuit, [encoding], [weights], observables

__all__ = ["build_classifier_circuit", "UnifiedQuantumClassifier"]
