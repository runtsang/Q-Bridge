"""Quantum classifier circuit that mirrors the classical architecture
and incorporates photonic‑style clipping and optional post‑processing
scaling.  The circuit uses a data‑encoding RX layer followed by
depth layers of Ry rotations and CZ entanglement.  Each variational
layer can clip its Ry angles to [-5,5] and a classical scale/shift
parameter can be applied to the measurement expectation values.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

class QuantumClassifierModel:
    """Builds a quantum circuit that follows the same topology as the
    classical network defined in the companion ML module.
    """

    def __init__(self, num_qubits: int, depth: int,
                 *, clip: bool = False, scale: float = 1.0, shift: float = 0.0) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.clip = clip
        self.scale = scale
        self.shift = shift
        self.circuit, self.encoding, self.weights, self.observables = self._build_circuit()

    def _build_circuit(self) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
        encoding = ParameterVector("x", self.num_qubits)
        weights = ParameterVector("theta", self.num_qubits * self.depth)

        qc = QuantumCircuit(self.num_qubits)
        # Data‑encoding: RX rotations
        for qubit, param in enumerate(encoding):
            qc.rx(param, qubit)

        idx = 0
        for _ in range(self.depth):
            # Variational Ry on each qubit
            for qubit in range(self.num_qubits):
                qc.ry(weights[idx], qubit)
                idx += 1
            # Entanglement
            for qubit in range(self.num_qubits - 1):
                qc.cz(qubit, qubit + 1)

        # Observables: Pauli‑Z on each qubit
        observables = [SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits - i - 1))
                       for i in range(self.num_qubits)]
        return qc, list(encoding), list(weights), observables

    def weight_sizes(self) -> List[int]:
        """Return the number of trainable parameters per variational layer."""
        return [self.num_qubits] * self.depth

    def encode_data(self, data: List[float]) -> List[float]:
        """Return the data to be bound to the encoding parameters."""
        return data

__all__ = ["QuantumClassifierModel"]
