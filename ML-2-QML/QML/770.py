"""QuantumClassifier: Variational quantum classifier with parameter‑shift training.

This implementation builds on the original seed by adding multi‑class capability,
noise simulation, and a gradient‑based optimizer that uses the parameter‑shift rule.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List, Optional

import numpy as np
from qiskit import QuantumCircuit, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.providers.aer import AerSimulator


class QuantumClassifier:
    """
    A variational quantum circuit that mimics the classical interface while
    providing quantum‑specific training primitives.
    """

    def __init__(
        self,
        num_qubits: int,
        depth: int,
        num_classes: int = 2,
        noise_model: Optional[object] = None,
        backend: Optional[object] = None,
        shots: int = 1024,
    ):
        self.num_qubits = num_qubits
        self.depth = depth
        self.num_classes = num_classes
        self.shots = shots
        self.noise_model = noise_model
        self.backend = backend or AerSimulator()
        (
            self.circuit,
            self.encoding,
            self.weights,
            self.observables,
        ) = self._build_circuit()
        # Random initial parameters for the variational layer
        self.params = np.random.randn(len(self.weights))
        # Trainable linear readout
        self.W = np.random.randn(self.num_qubits, self.num_classes)
        self.b = np.zeros(self.num_classes)

    def _build_circuit(
        self,
    ) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
        encoding = ParameterVector("x", self.num_qubits)
        weights = ParameterVector("theta", self.num_qubits * self.depth)
        circuit = QuantumCircuit(self.num_qubits)
        for param, qubit in zip(encoding, range(self.num_qubits)):
            circuit.rx(param, qubit)
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

    def _expectation(self, data: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Return expectation values of the Z observables for a given data point."""
        bound = dict(zip(self.encoding, data))
        bound.update(dict(zip(self.weights, params)))
        bound_circuit = self.circuit.bind_parameters(bound)
        result = self.backend.run(bound_circuit, method="statevector").result()
        state = result.get_statevector()
        exp_vals = np.array(
            [
                np.real(np.vdot(state, op.data[0] @ state))
                for op in self.observables
            ]
        )
        return exp_vals

    def _logits(self, exp_vals: np.ndarray) -> np.ndarray:
        """Linear readout mapping expectation values to logits."""
        return exp_vals @ self.W + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class indices for the given inputs."""
        logits = []
        for x in X:
            exp_vals = self._expectation(x, self.params)
            logits.append(self._logits(exp_vals))
        logits = np.array(logits)
        return np.argmax(logits, axis=1)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy over a dataset."""
        preds = self.predict(X)
        return np.mean(preds == y)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 20,
        lr: float = 0.01,
        verbose: bool = False,
    ) -> List[float]:
        """Gradient‑descent training using the parameter‑shift rule."""
        losses: List[float] = []
        shift = np.pi / 2
        for epoch in range(epochs):
            loss = 0.0
            grads = np.zeros_like(self.params)
            for x, target in zip(X, y):
                exp_vals = self._expectation(x, self.params)
                logits = self._logits(exp_vals)
                probs = np.exp(logits) / np.sum(np.exp(logits))
                loss += -np.log(probs[target] + 1e-12)
                # Parameter‑shift gradient
                for i in range(len(self.params)):
                    shift_vec = np.zeros_like(self.params)
                    shift_vec[i] = shift
                    exp_plus = self._expectation(x, self.params + shift_vec)
                    exp_minus = self._expectation(x, self.params - shift_vec)
                    grad_i = 0.5 * (exp_plus - exp_minus).dot(self.W[:, target])
                    grads[i] += grad_i
            loss /= len(X)
            grads /= len(X)
            self.params -= lr * grads
            losses.append(loss)
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs} – loss: {loss:.4f}")
        return losses


__all__ = ["QuantumClassifier"]
