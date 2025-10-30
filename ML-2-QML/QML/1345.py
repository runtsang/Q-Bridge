"""
Quantum neural network regressor.

A variational circuit with two trainable layers and a Pauli‑Z expectation readout.
The class implements a scikit‑learn compatible API with `fit` and `predict`.
"""
from __future__ import annotations

import pennylane as qml
import numpy as np
from pennylane import numpy as pnp
from pennylane.optimize import AdamOptimizer
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple


class EstimatorQNN(BaseEstimator, RegressorMixin):
    """
    Variational quantum circuit regressor.
    """

    def __init__(
        self,
        n_qubits: int = 1,
        n_layers: int = 2,
        lr: float = 0.1,
        epochs: int = 200,
        batch_size: int = 32,
    ) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = MinMaxScaler()
        self._build_circuit()

    def _build_circuit(self) -> None:
        dev = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(dev, interface="autograd")
        def circuit(inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
            for i in range(self.n_qubits):
                qml.RY(inputs[i], wires=i)
            for layer in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.RZ(weights[layer, i], wires=i)
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit
        self.weights = pnp.random.randn(self.n_layers, self.n_qubits)

    def _loss(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        preds = np.array([self.circuit(x, params) for x in X])
        return np.mean((preds - y) ** 2)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EstimatorQNN":
        X_np = X.astype(np.float64)
        y_np = y.astype(np.float64)
        X_scaled = self.scaler.fit_transform(X_np)
        opt = AdamOptimizer(self.lr)
        params = self.weights
        for _ in range(self.epochs):
            for i in range(0, len(X_scaled), self.batch_size):
                X_batch = X_scaled[i : i + self.batch_size]
                y_batch = y_np[i : i + self.batch_size]
                grads = opt.gradients(lambda w: self._loss(w, X_batch, y_batch), params)
                params = opt.step(grads, params)
        self.weights = params
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X.astype(np.float64))
        return np.array([self.circuit(x, self.weights) for x in X_scaled])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_qubits={self.n_qubits}, n_layers={self.n_layers})"


__all__ = ["EstimatorQNN"]
