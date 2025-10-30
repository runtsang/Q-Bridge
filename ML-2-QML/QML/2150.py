"""Hybrid variational classifier with quantum encoding and classical post‑processing.

The class exposes the same API as the classical counterpart:
* ``predict_proba`` returns class probabilities.
* ``predict`` returns the most likely class.
* ``compute_loss`` and ``accuracy`` are placeholders for future integration.

The circuit is a layered variational ansatz with:
* Data‑encoding rx rotations.
* Depth‑dependent ry rotations and CZ entangling gates.
* Classical readout via expectation values of Z on each qubit,
  followed by a sigmoid transform to obtain probabilities.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.providers.aer import AerSimulator


class QuantumClassifierModel:
    """
    A hybrid model that mimics the classical API using a qiskit circuit.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (features).
    depth : int
        Depth of the variational ansatz.
    backend : str, optional
        Name of the Aer simulator backend (default: 'aer_simulator_statevector').
    shots : int, optional
        Number of shots for measurement (default: 1024).
    """

    def __init__(
        self,
        num_qubits: int,
        depth: int,
        backend: str = "aer_simulator_statevector",
        shots: int = 1024,
    ) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.shots = shots
        self.backend = AerSimulator(method="statevector")
        self.circuit, self.encoding, self.weights, self.observables = self._build_circuit()

    def _build_circuit(
        self,
    ) -> Tuple[QuantumCircuit, Iterable, Iterable, list[SparsePauliOp]]:
        """Create a layered ansatz with encoding and variational parameters."""
        encoding = ParameterVector("x", self.num_qubits)
        weights = ParameterVector("theta", self.num_qubits * self.depth)

        qc = QuantumCircuit(self.num_qubits)

        # Data encoding
        for qubit, param in enumerate(encoding):
            qc.rx(param, qubit)

        # Variational layers
        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                qc.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(self.num_qubits - 1):
                qc.cz(qubit, qubit + 1)

        # Measurements are handled in predict_proba
        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits - i - 1))
            for i in range(self.num_qubits)
        ]
        return qc, list(encoding), list(weights), observables

    def _parametric_circuit(self, input_data: np.ndarray) -> QuantumCircuit:
        """Bind data and variational parameters to the circuit."""
        bound_circuit = self.circuit.bind_parameters(
            {name: val for name, val in zip(self.encoding, input_data)}
        )
        # Randomly initialise variational parameters if not set
        return bound_circuit

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return class probabilities for each sample.

        Parameters
        ----------
        X : np.ndarray
            Input matrix of shape (n_samples, num_qubits).

        Returns
        -------
        probs : np.ndarray
            Probabilities of shape (n_samples, 2).
        """
        probs = np.zeros((X.shape[0], 2))
        for i, sample in enumerate(X):
            qc = self._parametric_circuit(sample)
            job = self.backend.run(qc, shots=self.shots)
            result = job.result()
            counts = result.get_counts(qc)
            # Compute expectation of Z on each qubit
            exp_vals = np.array(
                [
                    (counts.get("1" * j + "0" * (self.num_qubits - j - 1), 0) -
                     counts.get("0" * j + "1" * (self.num_qubits - j - 1), 0))
                    / self.shots
                    for j in range(self.num_qubits)
                ]
            )
            # Map expectation values to probabilities via sigmoid
            probs[i, 0] = 1 / (1 + np.exp(-np.mean(exp_vals)))
            probs[i, 1] = 1 - probs[i, 0]
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class indices for each sample."""
        return np.argmax(self.predict_proba(X), axis=1)

    def compute_loss(self, logits: np.ndarray, y: np.ndarray) -> float:
        """Placeholder for cross‑entropy loss (classical)."""
        from scipy.special import softmax

        probs = softmax(logits, axis=1)
        eps = 1e-12
        loss = -np.mean(np.log(probs[np.arange(len(y)), y] + eps))
        return loss

    def accuracy(self, logits: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy over a batch."""
        preds = self.predict(logits)
        return np.mean(preds == y)

    @staticmethod
    def build_classifier_circuit(
        num_qubits: int, depth: int
    ) -> Tuple[QuantumCircuit, Iterable, Iterable, list[SparsePauliOp]]:
        """
        Factory that returns the same ansatz used in the class.

        Returns
        -------
        circuit : QuantumCircuit
            Constructed quantum circuit.
        encoding : Iterable
            List of ParameterVector names for data encoding.
        weights : Iterable
            List of ParameterVector names for variational weights.
        observables : list[SparsePauliOp]
            Z observables per qubit.
        """
        encoding = ParameterVector("x", num_qubits)
        weights = ParameterVector("theta", num_qubits * depth)

        qc = QuantumCircuit(num_qubits)
        for qubit, param in enumerate(encoding):
            qc.rx(param, qubit)

        idx = 0
        for _ in range(depth):
            for qubit in range(num_qubits):
                qc.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(num_qubits - 1):
                qc.cz(qubit, qubit + 1)

        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
            for i in range(num_qubits)
        ]
        return qc, list(encoding), list(weights), observables


__all__ = ["QuantumClassifierModel", "build_classifier_circuit"]
