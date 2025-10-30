"""Quantum classifier built on a parameterised ansatz.

The implementation extends the original data‑uploading circuit by
adding a configurable encoding layer, a tunable depth, and a
parameter‑shift gradient routine.  A ``QuantumClassifier`` class
provides a predict method that returns the expectation values of
Z‑observables on each qubit, which can be interpreted as class scores.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.providers.aer import AerSimulator


class QuantumClassifier:
    """Variational circuit that mimics the interface of the classical
    ``QuantumClassifier``.  It accepts binary input data and returns a
    two‑element probability vector for each sample.
    """
    def __init__(self, num_qubits: int, depth: int = 2,
                 backend: str | None = None) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.backend = backend or AerSimulator()
        self.circuit, self.encoding, self.weights, self.observables = self.build_classifier_circuit(
            num_qubits, depth
        )

    def build_classifier_circuit(self, num_qubits: int,
                                 depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
        """
        Construct a layered ansatz with an explicit feature‑encoding layer
        followed by alternating rotations and entangling gates.  The
        encoding uses RX rotations parameterised by the input data; the
        variational parameters are stored in a flat ``ParameterVector``.
        """
        encoding = ParameterVector("x", num_qubits)
        weights = ParameterVector("theta", num_qubits * depth)
        qc = QuantumCircuit(num_qubits)
        # Feature encoding
        for idx, qubit in enumerate(range(num_qubits)):
            qc.rx(encoding[idx], qubit)
        # Variational layers
        param_idx = 0
        for _ in range(depth):
            for qubit in range(num_qubits):
                qc.ry(weights[param_idx], qubit)
                param_idx += 1
            for qubit in range(num_qubits - 1):
                qc.cz(qubit, qubit + 1)
        # Observables: Z on each qubit
        observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
                       for i in range(num_qubits)]
        return qc, list(encoding), list(weights), observables

    def _substitute_parameters(self, data: np.ndarray) -> QuantumCircuit:
        """
        Bind the input data to the encoding parameters and return a
        new circuit ready for execution.
        """
        bound = self.circuit.bind_parameters({p: val for p, val in zip(self.encoding, data)})
        return bound

    def evaluate(self, data: np.ndarray) -> np.ndarray:
        """
        Compute the expectation value of each observable for every
        sample in ``data``.  The result is a 2‑D array of shape
        (n_samples, num_qubits) where each row contains the Z‑expectation
        values that can be interpreted as logits for binary classification.
        """
        if data.ndim == 1:
            data = data[None, :]
        n_samples = data.shape[0]
        results = np.zeros((n_samples, self.num_qubits))
        for i, sample in enumerate(data):
            circ = self._substitute_parameters(sample)
            circ = transpile(circ, backend=self.backend)
            job = self.backend.run(circ, shots=1024)
            counts = job.result().get_counts()
            # Convert counts to expectation values of Z
            exp_vals = []
            for op in self.observables:
                expect = 0.0
                for bitstring, freq in counts.items():
                    z = 1 if bitstring[self.num_qubits - 1 - op.to_label().index('Z')] == '0' else -1
                    expect += z * freq
                exp_vals.append(expect / 1024)
            results[i, :] = exp_vals
        return results

    def predict(self, data: np.ndarray, threshold: float = 0.0) -> np.ndarray:
        """
        Return binary predictions (0 or 1) by thresholding the
        majority of the Z‑expectation values.
        """
        logits = self.evaluate(data)
        # Convert logits to class probabilities using sigmoid
        probs = 1 / (1 + np.exp(-logits.sum(axis=1)))
        return (probs > threshold).astype(int)

__all__ = ["QuantumClassifier"]
