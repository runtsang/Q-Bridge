"""Hybrid classifier class with a variational quantum circuit."""
from __future__ import annotations

from typing import Iterable, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


class QuantumClassifierModel:
    """
    Variational circuit that parallels the classical network interface.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (features).
    depth : int
        Depth of the ansatz.
    entanglement : str, default 'cnot'
        Entanglement pattern; options: 'cnot', 'cz', 'identity'.
    measurement : str, default 'PauliZ'
        Observable type for each qubit.
    """

    def __init__(
        self,
        num_qubits: int,
        depth: int,
        entanglement: str = "cnot",
        measurement: str = "PauliZ",
    ) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.entanglement = entanglement
        self.measurement = measurement
        (
            self.circuit,
            self.encoding,
            self.weights,
            self.observables,
        ) = self._build_circuit()
        self.weight_sizes = [len(self.weights)]

    def _build_circuit(self) -> tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
        """Constructs the variational ansatz."""
        encoding = ParameterVector("x", self.num_qubits)
        weights = ParameterVector("theta", self.num_qubits * self.depth)

        qc = QuantumCircuit(self.num_qubits)

        # Feature encoding
        for qubit, param in enumerate(encoding):
            qc.rx(param, qubit)

        # Variational layers
        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                qc.ry(weights[idx], qubit)
                idx += 1
            self._apply_entanglement(qc)

        # Observables
        if self.measurement == "PauliZ":
            observables = [
                SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits - i - 1))
                for i in range(self.num_qubits)
            ]
        else:
            observables = [SparsePauliOp("I" * self.num_qubits)]  # dummy

        return qc, list(encoding), list(weights), observables

    def _apply_entanglement(self, qc: QuantumCircuit) -> None:
        """Adds entanglement gates according to the chosen pattern."""
        if self.entanglement == "cnot":
            for qubit in range(self.num_qubits - 1):
                qc.cx(qubit, qubit + 1)
        elif self.entanglement == "cz":
            for qubit in range(self.num_qubits - 1):
                qc.cz(qubit, qubit + 1)
        # 'identity' leaves qubits unentangled

    def get_encoding(self) -> List[ParameterVector]:
        """Return the encoding parameters."""
        return self.encoding

    def get_weight_sizes(self) -> List[int]:
        """Return the number of variational parameters."""
        return self.weight_sizes

    def get_observables(self) -> List[SparsePauliOp]:
        """Return the list of measurement operators."""
        return self.observables

__all__ = ["QuantumClassifierModel"]
