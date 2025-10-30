"""Variational quantum classifier built with Pennylane.

The class implements a data‑encoding layer followed by a depth‑controlled
ansatz of RY rotations and CZ entangling gates.  Training uses
parameter‑shift gradient descent, and the API mirrors the classical
counterpart: fit, predict and predict_proba.
"""

from __future__ import annotations

import pennylane as qml
import numpy as np
from typing import Iterable

class QuantumClassifierModel:
    """
    Quantum classifier that mimics the classical API.  It trains a
    variational circuit on a user‑supplied dataset and outputs class
    predictions or probability estimates.
    """
    def __init__(
        self,
        num_qubits: int,
        depth: int = 2,
        shots: int = 1024,
        lr: float = 0.01,
        epochs: int = 50,
    ):
        self.num_qubits = num_qubits
        self.depth = depth
        self.shots = shots
        self.lr = lr
        self.epochs = epochs
        self.dev = qml.device("default.qubit", wires=num_qubits, shots=shots)
        self.params = np.random.randn(num_qubits * depth)
        self.qnode = qml.QNode(self._circuit, self.dev)

    def _circuit(self, x: Iterable[float], params: np.ndarray) -> list[float]:
        # Data‑encoding (RX rotations)
        for i, val in enumerate(x):
            qml.RX(val, wires=i)
        # Variational ansatz
        idx = 0
        for _ in range(self.depth):
            for i in range(self.num_qubits):
                qml.RY(params[idx], wires=i)
                idx += 1
            for i in range(self.num_qubits - 1):
                qml.CZ(wires=[i, i + 1])
        # Measurement of Z on each qubit
        return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

    def _loss(self, params: np.ndarray, x: Iterable[float], y: int) -> float:
        preds = self._circuit(x, params)
        # Use first two expectation values as logits
        logits = np.array(preds[:2])
        probs = np.exp(logits) / np.sum(np.exp(logits))
        return -np.log(probs[y] + 1e-9)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        for _ in range(self.epochs):
            epoch_loss = 0.0
            for xi, yi in zip(X, y):
                grad = qml.grad(self._loss)(self.params, xi, yi)
                self.params -= self.lr * grad
                epoch_loss += self._loss(self.params, xi, yi)

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = []
        for xi in X:
            logits = np.array(self._circuit(xi, self.params)[:2])
            preds.append(np.argmax(logits))
        return np.array(preds)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs_list = []
        for xi in X:
            logits = np.array(self._circuit(xi, self.params)[:2])
            probs = np.exp(logits) / np.sum(np.exp(logits))
            probs_list.append(probs)
        return np.array(probs_list)

__all__ = ["QuantumClassifierModel"]
