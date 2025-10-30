"""Quantum classifier with data‑re‑uploading and parameter‑shift training.

This module defines :class:`QuantumClassifierModel` – a variational circuit that
mirrors the original ``build_classifier_circuit`` helper but exposes a richer
training API.  It supports configurable depth, optional data‑re‑uploading,
and a flexible backend interface.  The class implements a simple
parameter‑shift gradient descent method and returns the expectation values
of a Pauli‑Z observable for each qubit.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


class QuantumClassifierModel:
    """
    Variational classifier for 2‑class problems.

    Parameters
    ----------
    num_qubits : int
        Number of qubits / dimensionality of the input data.
    depth : int, default 2
        Number of variational layers.
    reupload : bool, default True
        If ``True`` the data is encoded before every layer (data‑re‑uploading).
    shots : int, default 1024
        Number of shots used when sampling expectation values.
    device : str, default 'qasm_simulator'
        Backend name to be fetched from :class:`qiskit.Aer`.
    """

    def __init__(
        self,
        num_qubits: int,
        depth: int = 2,
        reupload: bool = True,
        shots: int = 1024,
        device: str = "qasm_simulator",
    ) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.reupload = reupload
        self.shots = shots
        self.backend = Aer.get_backend(device)

        (
            self.circuit,
            self.encoding,
            self.weights,
            self.observables,
        ) = self._build_circuit()
        self.params = np.random.randn(len(self.weights))

    # ------------------------------------------------------------------
    # Circuit construction
    # ------------------------------------------------------------------
    def _build_circuit(
        self,
    ) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
        """Return a circuit, encoding parameters, variational parameters and observables."""
        encoding = ParameterVector("x", self.num_qubits)
        weights = ParameterVector("theta", self.num_qubits * self.depth)
        qc = QuantumCircuit(self.num_qubits)

        if self.reupload:
            for layer in range(self.depth):
                # data encoding
                for qubit, param in enumerate(encoding):
                    qc.rx(param, qubit)
                # variational rotation
                for idx, w in enumerate(
                    weights[self.num_qubits * layer : self.num_qubits * (layer + 1)]
                ):
                    qc.ry(w, idx)
                qc.barrier()
                # entangling layer
                for qubit in range(self.num_qubits - 1):
                    qc.cz(qubit, qubit + 1)
        else:
            # single encoding
            for qubit, param in enumerate(encoding):
                qc.rx(param, qubit)
            # variational rotations
            for idx, w in enumerate(weights):
                qc.ry(w, idx)
            # entangling layer
            for qubit in range(self.num_qubits - 1):
                qc.cz(qubit, qubit + 1)

        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits - i - 1))
            for i in range(self.num_qubits)
        ]

        return qc, list(encoding), list(weights), observables

    # ------------------------------------------------------------------
    # Forward / inference
    # ------------------------------------------------------------------
    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Compute expectation values for each observable.

        Parameters
        ----------
        data : np.ndarray, shape (n_samples, num_qubits)
            Classical input vectors.

        Returns
        -------
        np.ndarray, shape (n_samples, num_qubits)
            Expectation values of the Z observables for each sample.
        """
        results = []

        for sample in data:
            bound = self.circuit.bind_parameters(
                {p: val for p, val in zip(self.encoding, sample)}
            )
            job = execute(bound, self.backend, shots=self.shots)
            result = job.result()
            counts = result.get_counts()
            expectations = []

            for obs in self.observables:
                exp = 0.0
                # Pauli‑Z observable acts on qubit ``i`` where ``i`` is the position
                # of the 'Z' in the string.
                qubit_index = obs.paulis[0][0]
                for bitstring, count in counts.items():
                    # Qiskit returns bit strings in little‑endian order.
                    bit = bitstring[::-1][qubit_index]
                    exp += (1 if bit == "0" else -1) * count
                exp /= self.shots
                expectations.append(exp)

            results.append(expectations)

        return np.array(results)

    # ------------------------------------------------------------------
    # Training utilities
    # ------------------------------------------------------------------
    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def loss_fn(self, preds: np.ndarray, y: np.ndarray) -> float:
        """Cross‑entropy loss between predictions and one‑hot labels."""
        probs = self.softmax(preds)
        eps = 1e-12
        return -np.mean(np.log(probs[np.arange(len(y)), y] + eps))

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr: float = 0.01,
        epochs: int = 10,
    ) -> None:
        """
        Gradient‑descent training using the parameter‑shift rule.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, num_qubits)
            Training data.
        y : np.ndarray, shape (n_samples,)
            Integer class labels (0 or 1).
        lr : float, default 0.01
            Learning rate.
        epochs : int, default 10
            Number of training epochs.
        """
        for epoch in range(epochs):
            preds = self.predict(X)
            loss = self.loss_fn(preds, y)
            grads = self._parameter_shift_gradients(X, y)
            self.params -= lr * grads
            self._update_circuit_params()

    def _parameter_shift_gradients(
        self, X: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """Estimate gradients via the parameter‑shift rule."""
        grads = np.zeros_like(self.params)
        shift = np.pi / 2

        for i in range(len(self.params)):
            params_plus = self.params.copy()
            params_minus = self.params.copy()
            params_plus[i] += shift
            params_minus[i] -= shift

            self._set_params(params_plus)
            loss_plus = self.loss_fn(self.predict(X), y)

            self._set_params(params_minus)
            loss_minus = self.loss_fn(self.predict(X), y)

            grads[i] = (loss_plus - loss_minus) / (2 * np.sin(shift))

        return grads

    def _set_params(self, params: np.ndarray) -> None:
        """Assign a new set of parameters to the variational circuit."""
        for i, w in enumerate(self.weights):
            self.circuit.assign_parameters({w: params[i]}, inplace=True)
        self.params = params

    def _update_circuit_params(self) -> None:
        """Synchronise the internal parameter vector with the circuit."""
        self._set_params(self.params)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    @staticmethod
    def build_classifier_circuit(
        num_qubits: int,
        depth: int,
        reupload: bool = True,
        shots: int = 1024,
        device: str = "qasm_simulator",
    ) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
        """
        Factory that returns a freshly constructed circuit and its metadata.
        """
        model = QuantumClassifierModel(num_qubits, depth, reupload, shots, device)
        return (
            model.circuit,
            model.encoding,
            model.weights,
            model.observables,
        )


__all__ = ["QuantumClassifierModel"]
