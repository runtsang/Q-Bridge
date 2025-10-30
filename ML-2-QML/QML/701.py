"""Quantum classifier factory with configurable entanglement and encoding."""

from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


class QuantumClassifierModel:
    """A parameter‑driven quantum circuit that mirrors the classical factory.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    depth : int
        Number of variational layers.
    entanglement_pattern : str, optional
        Either ``"linear"`` or ``"circular"``; determines CZ connectivity.
    data_encoding : str, optional
        Either ``"rx"`` or ``"ry"``; selects the single‑qubit data‑encoding gate.
    """

    def __init__(
        self,
        num_qubits: int,
        depth: int,
        entanglement_pattern: str = "linear",
        data_encoding: str = "rx",
    ) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.entanglement_pattern = entanglement_pattern
        self.data_encoding = data_encoding

        self.encoding_params = ParameterVector("x", num_qubits)
        self.weight_params = ParameterVector("theta", num_qubits * depth)

        self.circuit = self._build_circuit()

        # Metadata buffers
        self.encoding_indices = list(range(num_qubits))
        self.observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
            for i in range(num_qubits)
        ]

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)

        # Data encoding
        if self.data_encoding == "rx":
            for qubit, param in zip(range(self.num_qubits), self.encoding_params):
                qc.rx(param, qubit)
        elif self.data_encoding == "ry":
            for qubit, param in zip(range(self.num_qubits), self.encoding_params):
                qc.ry(param, qubit)
        else:
            raise ValueError(f"Unsupported data encoding: {self.data_encoding}")

        # Ansatz layers
        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                qc.ry(self.weight_params[idx], qubit)
                idx += 1
            if self.entanglement_pattern == "linear":
                for qubit in range(self.num_qubits - 1):
                    qc.cz(qubit, qubit + 1)
            elif self.entanglement_pattern == "circular":
                for qubit in range(self.num_qubits):
                    qc.cz(qubit, (qubit + 1) % self.num_qubits)
            else:
                raise ValueError(f"Unsupported entanglement: {self.entanglement_pattern}")

        return qc

    def get_circuit(self) -> QuantumCircuit:
        """Return the fully constructed quantum circuit."""
        return self.circuit

    def get_encoding_parameters(self) -> ParameterVector:
        """Return the data‑encoding parameter vector."""
        return self.encoding_params

    def get_weight_parameters(self) -> ParameterVector:
        """Return the variational weight parameter vector."""
        return self.weight_params

    def get_observables(self) -> List[SparsePauliOp]:
        """Return the list of measurement observables."""
        return self.observables

    def get_encoding_indices(self) -> List[int]:
        """Return the indices of qubits used for data encoding."""
        return self.encoding_indices


def build_classifier_circuit(
    num_qubits: int,
    depth: int,
    entanglement_pattern: str = "linear",
    data_encoding: str = "rx",
) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """Construct a quantum classifier circuit and return metadata.

    Returns
    -------
    circuit : QuantumCircuit
        The constructed Qiskit circuit.
    encoding : Iterable
        Indices of qubits used for data encoding.
    weights : Iterable
        ParameterVector containing all variational parameters.
    observables : List[SparsePauliOp]
        Measurement observables for each qubit.
    """
    model = QuantumClassifierModel(
        num_qubits, depth, entanglement_pattern, data_encoding
    )
    return (
        model.get_circuit(),
        model.get_encoding_indices(),
        model.get_weight_parameters(),
        model.get_observables(),
    )


__all__ = ["QuantumClassifierModel", "build_classifier_circuit"]
