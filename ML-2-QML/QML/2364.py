"""QuantumClassifierModel: Variational circuit with data‑uploading ansatz.

The class builds a parameterised quantum circuit that mirrors the classical
network metadata.  It exposes the same ``encoding`` and ``weight_sizes`` fields
so that a hybrid training loop can use identical descriptors for both
components.  The circuit uses a simple RX data‑encoding followed by a
depth‑controlled sequence of RY rotations and CZ entangling gates.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


class QuantumClassifierModel:
    """Variational quantum classifier.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (equal to the number of input features).
    depth : int
        Number of variational layers.
    """

    def __init__(self, num_qubits: int, depth: int) -> None:
        self.num_qubits = num_qubits
        self.depth = depth

        # Parameter vectors
        self.encoding = ParameterVector("x", num_qubits)
        self.weights = ParameterVector("theta", num_qubits * depth)

        # Build circuit
        self.circuit = self._build_circuit()

        # Metadata
        self.weight_sizes = [num_qubits] * depth  # one RY per qubit per layer
        self.observables = self._build_observables()

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)

        # Data‑encoding layer
        for idx, qubit in enumerate(range(self.num_qubits)):
            qc.rx(self.encoding[idx], qubit)

        # Variational layers
        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                qc.ry(self.weights[idx], qubit)
                idx += 1
            for qubit in range(self.num_qubits - 1):
                qc.cz(qubit, qubit + 1)

        return qc

    def _build_observables(self) -> List[SparsePauliOp]:
        """Return a list of Z‑observables on each qubit."""
        return [
            SparsePauliOp.from_list([("I" * i + "Z" + "I" * (self.num_qubits - i - 1), 1)])
            for i in range(self.num_qubits)
        ]

    @property
    def encoding_meta(self) -> Iterable[ParameterVector]:
        """Return the encoding parameters."""
        return [self.encoding]

    @property
    def weight_meta(self) -> Iterable[int]:
        """Return the number of variational parameters per layer."""
        return self.weight_sizes

    @property
    def observable_meta(self) -> Iterable[SparsePauliOp]:
        """Return the list of observables."""
        return self.observables


def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumClassifierModel, Iterable[ParameterVector], Iterable[int], List[SparsePauliOp]]:
    """Factory that returns a ``QuantumClassifierModel`` instance and its metadata.

    Parameters
    ----------
    num_qubits : int
        Number of qubits / input dimensionality.
    depth : int
        Number of variational layers.

    Returns
    -------
    model : QuantumClassifierModel
        The instantiated variational circuit.
    encoding : Iterable[ParameterVector]
        Parameter vectors used for data encoding.
    weight_sizes : Iterable[int]
        Number of variational parameters per layer.
    observables : List[SparsePauliOp]
        Z‑observables on each qubit.
    """
    model = QuantumClassifierModel(num_qubits, depth)
    return model, model.encoding_meta, model.weight_meta, model.observable_meta


__all__ = ["QuantumClassifierModel", "build_classifier_circuit"]
