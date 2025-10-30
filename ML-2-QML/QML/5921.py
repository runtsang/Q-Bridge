"""Quantum classifier circuit factory with data re-uploading and entanglement."""

from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.providers.aer import AerSimulator
import numpy as np


class QuantumClassifierModel:
    """
    A data‑re‑uploading variational classifier that mirrors the classical API.

    Enhancements over the seed:
    - Entangling layer between each data‑encoding step.
    - Flexible depth and qubit count.
    - A built-in simulator for expectation evaluation.
    - Metadata surfaces: encoding parameters, variational parameters, and observables.
    """

    def __init__(self, num_qubits: int, depth: int) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.circuit, self.encoding, self.weights, self.observables = self._build_circuit()
        self.simulator = AerSimulator()

    def _build_circuit(self) -> Tuple[QuantumCircuit, List[ParameterVector],
                                      List[ParameterVector], List[SparsePauliOp]]:
        """
        Build a layered ansatz with data re‑uploading and CZ entanglement.
        """
        encoding = ParameterVector("x", self.num_qubits)
        weights = ParameterVector("theta", self.num_qubits * self.depth)
        circuit = QuantumCircuit(self.num_qubits)

        # initial data encoding
        for q, param in enumerate(encoding):
            circuit.rx(param, q)

        idx = 0
        for _ in range(self.depth):
            # variational rotation
            for q in range(self.num_qubits):
                circuit.ry(weights[idx], q)
                idx += 1
            # entanglement
            for q in range(self.num_qubits - 1):
                circuit.cz(q, q + 1)
            # re‑upload data
            for q, param in enumerate(encoding):
                circuit.rx(param, q)

        # observables: Pauli‑Z on each qubit
        observables = [SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits - i - 1))
                       for i in range(self.num_qubits)]
        return circuit, list(encoding), list(weights), observables

    def evaluate(self, data: np.ndarray, params: np.ndarray) -> np.ndarray:
        """
        Compute expectation values of the observables for a batch of inputs.

        Parameters
        ----------
        data : np.ndarray
            Shape (batch, num_qubits) — classical feature vectors.
        params : np.ndarray
            Shape (num_params,) — variational parameters.

        Returns
        -------
        np.ndarray
            Shape (batch, num_qubits) — expectation values per observable.
        """
        batch_size = data.shape[0]
        exp_vals = np.zeros((batch_size, self.num_qubits))
        for i in range(batch_size):
            bound = self.circuit.bind_parameters(
                {**{p: val for p, val in zip(self.encoding, data[i])},
                 **{w: val for w, val in zip(self.weights, params)}}
            )
            result = self.simulator.compute_expectation_values(bound, self.observables,
                                                               statevector=False)
            exp_vals[i] = result.real
        return exp_vals

    @staticmethod
    def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[
        QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
        """
        Factory that mirrors the classical helper signature.
        """
        model = QuantumClassifierModel(num_qubits, depth)
        return model.circuit, model.encoding, model.weights, model.observables


__all__ = ["QuantumClassifierModel"]
