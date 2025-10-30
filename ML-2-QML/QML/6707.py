"""Quantum classifier using a data‑re‑uploading variational ansatz.

The implementation follows the original design but adds a simple numerical
gradient routine for parameter updates, a sigmoid transform to obtain
probabilities, and a convenient ``predict`` method.  The class keeps the
same public attributes (encoding, weights, observables) so that it can be
used interchangeably with the classical counterpart in experimental
pipelines.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, SparsePauliOp


class QuantumClassifierModel:
    """
    Variational quantum classifier with data‑re‑uploading ansatz.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (equal to the input dimensionality).
    depth : int
        Number of variational layers.
    backend : str, optional
        Backend name for state‑vector simulation.
    shots : int, optional
        Number of shots for sampling (unused in state‑vector mode).
    """

    def __init__(
        self,
        num_qubits: int,
        depth: int,
        backend: str = "statevector_simulator",
        shots: int = 1024,
    ):
        self.num_qubits = num_qubits
        self.depth = depth
        self.backend = backend
        self.shots = shots

        (
            self.circuit,
            self.encoding,
            self.weights,
            self.observables,
        ) = self._build_circuit()

        # Initialise all parameters to zeros for simplicity
        self.params = np.concatenate([self.encoding, self.weights])

    def _build_circuit(
        self,
    ) -> Tuple[QuantumCircuit, Iterable, Iterable, list[SparsePauliOp]]:
        """
        Build a layered ansatz with explicit data encoding and variational layers.
        """
        encoding = ParameterVector("x", self.num_qubits)
        weights = ParameterVector("theta", self.num_qubits * self.depth)

        circuit = QuantumCircuit(self.num_qubits)

        # Data encoding with RX rotations
        for i in range(self.num_qubits):
            circuit.rx(encoding[i], i)

        # Variational layers
        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                circuit.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(self.num_qubits - 1):
                circuit.cz(qubit, qubit + 1)

        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits - i - 1))
            for i in range(self.num_qubits)
        ]

        return circuit, list(encoding), list(weights), observables

    # --------------------------------------------------------------------- #
    # Core evaluation helpers
    # --------------------------------------------------------------------- #
    def expectation(self, data: np.ndarray) -> np.ndarray:
        """
        Compute expectation values of the Z observables for each sample.

        Parameters
        ----------
        data : np.ndarray
            Shape (n_samples, num_qubits).

        Returns
        -------
        np.ndarray
            Shape (n_samples, num_qubits) with expectation values.
        """
        results = []
        for sample in data:
            circ = self.circuit.copy()
            bind_dict = {p: sample[i] for i, p in enumerate(self.encoding)}
            for w in self.weights:
                bind_dict[w] = 0.0
            circ = circ.bind_parameters(bind_dict)
            state = Statevector.from_instruction(circ)
            results.append([state.expectation_value(obs).real for obs in self.observables])
        return np.array(results)

    # --------------------------------------------------------------------- #
    # Loss / training helpers
    # --------------------------------------------------------------------- #
    def _loss_with_weights(self, data: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> float:
        """
        Compute a binary cross‑entropy loss after applying a sigmoid to the
        expectation values.

        Parameters
        ----------
        data : np.ndarray
            Input samples, shape (n_samples, num_qubits).
        labels : np.ndarray
            Binary labels, shape (n_samples,).
        weights : np.ndarray
            Current variational parameters.

        Returns
        -------
        float
            Loss value.
        """
        # Bind weight parameters
        bind_dict_weights = {w: weights[i] for i, w in enumerate(self.weights)}

        results = []
        for sample in data:
            circ = self.circuit.copy()
            # data binding
            circ = circ.bind_parameters({p: sample[i] for i, p in enumerate(self.encoding)})
            # weight binding
            circ = circ.bind_parameters(bind_dict_weights)
            state = Statevector.from_instruction(circ)
            results.append([state.expectation_value(obs).real for obs in self.observables])

        exp_vals = np.array(results)  # shape (n_samples, num_qubits)
        probs = 1.0 / (1.0 + np.exp(-exp_vals))  # sigmoid per qubit
        # Average over qubits to obtain a single probability per sample
        probs = probs.mean(axis=1)
        loss = -np.mean(
            labels * np.log(probs + 1e-12) + (1.0 - labels) * np.log(1.0 - probs + 1e-12)
        )
        return loss

    def train_step(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        lr: float = 0.01,
        epsilon: float = 1e-5,
    ) -> float:
        """
        Perform one gradient‑descent step using a central finite‑difference
        approximation for the gradient.

        Parameters
        ----------
        data : np.ndarray
            Training samples, shape (n_samples, num_qubits).
        labels : np.ndarray
            Binary labels, shape (n_samples,).
        lr : float, optional
            Learning rate.
        epsilon : float, optional
            Perturbation size for finite‑difference.

        Returns
        -------
        float
            Updated loss value.
        """
        grads = np.zeros_like(self.weights)
        for i in range(len(self.weights)):
            w_plus = self.weights.copy()
            w_minus = self.weights.copy()
            w_plus[i] += epsilon
            w_minus[i] -= epsilon
            loss_plus = self._loss_with_weights(data, labels, w_plus)
            loss_minus = self._loss_with_weights(data, labels, w_minus)
            grads[i] = (loss_plus - loss_minus) / (2.0 * epsilon)

        self.weights -= lr * grads
        return self._loss_with_weights(data, labels, self.weights)

    # --------------------------------------------------------------------- #
    # Prediction helpers
    # --------------------------------------------------------------------- #
    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Predict class labels for the given data.

        Parameters
        ----------
        data : np.ndarray
            Input samples, shape (n_samples, num_qubits).

        Returns
        -------
        np.ndarray
            Predicted class indices (0 or 1).
        """
        exp_vals = self.expectation(data)
        probs = 1.0 / (1.0 + np.exp(-exp_vals))
        probs = probs.mean(axis=1)
        return (probs >= 0.5).astype(int)

__all__ = ["QuantumClassifierModel"]
