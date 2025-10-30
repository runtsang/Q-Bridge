"""QuantumClassifierModel – Quantum implementation using Pennylane.

This class implements a variational quantum classifier with amplitude
encoding and a single Pauli‑Z measurement.  It follows the same
interface as the classical counterpart (fit, predict, accuracy) so
they can be interchanged in pipelines.

Usage example:

    import numpy as np
    from QuantumClassifierModel__gen526 import QuantumClassifierModel

    model = QuantumClassifierModel(num_qubits=5, depth=3)
    model.fit(X_train, y_train, epochs=200)
    preds = model.predict(X_test)
"""

from __future__ import annotations

from typing import Iterable, Tuple

import pennylane as qml
import pennylane.numpy as np
import torch
import torch.nn as nn
from pennylane import GradientDescentOptimizer


class QuantumClassifierModel:
    """Variational quantum classifier using amplitude encoding."""

    def __init__(
        self,
        num_qubits: int,
        depth: int,
        lr: float = 0.01,
        device: str = "cpu",
    ) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.device = device

        self.dev = qml.device("default.qubit", wires=self.num_qubits)
        self.params = self._initialize_params()
        self.criterion = nn.BCEWithLogitsLoss()

    def _initialize_params(self) -> np.ndarray:
        """Randomly initialise variational parameters."""
        return np.random.randn(self.num_qubits * self.depth)

    def _circuit(self, features: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Variational ansatz with amplitude encoding."""
        qml.AmplitudeEmbedding(
            features=features,
            wires=range(self.num_qubits),
            normalize=True,
        )
        idx = 0
        for _ in range(self.depth):
            for q in range(self.num_qubits):
                qml.RY(params[idx], wires=q)
                idx += 1
            for q in range(self.num_qubits - 1):
                qml.CZ(wires=[q, q + 1])
        return qml.expval(qml.PauliZ(0))

    self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")

    def _cost(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> torch.Tensor:
        """Binary cross‑entropy cost for a batch of examples."""
        probs = []
        for x in X:
            out = self.qnode(x, params)
            probs.append(torch.sigmoid(torch.tensor(out)))
        probs = torch.stack(probs)
        loss = self.criterion(probs, torch.tensor(y, dtype=torch.float32))
        return loss

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 200,
        lr: float = 0.01,
        verbose: bool = True,
    ) -> None:
        """Train the variational circuit."""
        opt = GradientDescentOptimizer(stepsize=lr)
        params = self.params
        for epoch in range(1, epochs + 1):
            params, loss = opt.step_and_cost(
                lambda p: self._cost(p, X, y), params
            )
            self.params = params
            if verbose and epoch % max(1, epochs // 10) == 0:
                acc = self.accuracy(X, y)
                print(
                    f"Epoch {epoch}/{epochs} - loss: {loss:.4f} - acc: {acc:.4f}"
                )

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return class predictions for X."""
        preds = []
        for x in X:
            out = self.qnode(x, self.params)
            prob = torch.sigmoid(torch.tensor(out)).item()
            preds.append(1 if prob >= threshold else 0)
        return np.array(preds)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute classification accuracy."""
        preds = self.predict(X)
        return np.mean(preds == y)

    def get_parameters(self) -> np.ndarray:
        """Return the current parameter vector."""
        return self.params

    def __repr__(self) -> str:
        return (
            f"QuantumClassifierModel(num_qubits={self.num_qubits}, "
            f"depth={self.depth})"
        )


__all__ = ["QuantumClassifierModel"]
