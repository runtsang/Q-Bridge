"""Quantum classifier using a data‑re‑uploading variational ansatz."""

from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
import numpy as np


class QuantumClassifierModel:
    """
    Variational quantum classifier that emulates the classical interface.
    Parameters
    ----------
    num_qubits : int
        Number of qubits (feature dimension).
    depth : int
        Number of variational layers.
    """

    def __init__(self, num_qubits: int, depth: int) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.circuit, self.encoding, self.weights, self.observables = self._build_circuit()

    def _build_circuit(self) -> Tuple[QuantumCircuit, Iterable[ParameterVector], Iterable[ParameterVector], List[SparsePauliOp]]:
        """Create a data‑re‑uploading ansatz with alternating RX/RY rotations and CZ entangling gates."""
        encoding = ParameterVector("x", self.num_qubits)
        weights = ParameterVector("theta", self.num_qubits * self.depth)

        qc = QuantumCircuit(self.num_qubits)

        # Feature encoding
        for qubit, param in zip(range(self.num_qubits), encoding):
            qc.rx(param, qubit)

        # Variational layers
        idx = 0
        for _ in range(self.depth):
            # Rotation layer
            for qubit in range(self.num_qubits):
                qc.ry(weights[idx], qubit)
                idx += 1
            # Entangling layer (full‑chain CZ)
            for qubit in range(self.num_qubits - 1):
                qc.cz(qubit, qubit + 1)

        # Observables – single‑qubit Pauli‑Z measurements
        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits - i - 1))
            for i in range(self.num_qubits)
        ]

        return qc, encoding, weights, observables

    def expectation(self, data: np.ndarray, params: np.ndarray) -> np.ndarray:
        """
        Compute expectation values of the observables for a batch of inputs.
        Parameters
        ----------
        data : np.ndarray, shape (batch, num_qubits)
            Classical feature vectors.
        params : np.ndarray, shape (num_params,)
            Parameter values for the variational circuit.
        Returns
        -------
        np.ndarray, shape (batch, num_qubits)
            Expectation values of each observable.
        """
        # For brevity, use the statevector simulator
        from qiskit.quantum_info import Statevector

        batch_expectations = []
        for sample in data:
            bound_qc = self.circuit.bind_parameters(
                dict(zip(self.encoding, sample.tolist())) |
                dict(zip(self.weights, params.tolist()))
            )
            state = Statevector.from_instruction(bound_qc)
            exps = [float(state.expectation_value(op)) for op in self.observables]
            batch_expectations.append(exps)
        return np.array(batch_expectations)

    @staticmethod
    def build_classifier_circuit(
        num_qubits: int, depth: int
    ) -> Tuple[QuantumCircuit, Iterable[int], Iterable[int], List[SparsePauliOp]]:
        """
        Compatibility wrapper mirroring the classical API.
        Returns the raw circuit and metadata.
        """
        instance = QuantumClassifierModel(num_qubits, depth)
        return instance.circuit, list(range(num_qubits)), list(range(num_qubits * depth)), instance.observables


__all__ = ["QuantumClassifierModel"]
