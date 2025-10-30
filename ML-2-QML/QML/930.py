"""Quantum circuit and classifier for QuantumClassifierModel."""

from __future__ import annotations

import pennylane as qml
import numpy as np
from typing import Iterable, Tuple, List

class QuantumClassifierModel:
    """Variational quantum classifier with entangling layers and measurement-based loss."""
    def __init__(self, num_qubits: int, depth: int, device: str = "default.qubit"):
        self.num_qubits = num_qubits
        self.depth = depth
        self.dev = qml.device(device, wires=num_qubits)
        self.weights = np.random.randn(depth, num_qubits)
        self._build_circuit()

    def _build_circuit(self):
        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs, weights):
            # feature map
            for i, w in enumerate(inputs):
                qml.RX(w, i)
            # variational layers
            for layer in range(self.depth):
                for i in range(self.num_qubits):
                    qml.RY(weights[layer, i], i)
                for i in range(self.num_qubits - 1):
                    qml.CZ(i, i + 1)
            # measurement
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]
        self.circuit = circuit

    def loss(self, params, batch, labels):
        preds = []
        for x in batch:
            out = self.circuit(x, params)
            preds.append(out)
        preds = np.array(preds)
        # simple cross-entropy on sigmoid of measurement
        logits = preds[:, 0]  # pick first qubit as class 0
        probs = 1 / (1 + np.exp(-logits))
        loss = -np.mean(labels * np.log(probs + 1e-12) + (1 - labels) * np.log(1 - probs + 1e-12))
        return loss

    def fit(self, train_loader, epochs: int = 10, lr: float = 0.01):
        opt = qml.GradientDescentOptimizer(stepsize=lr)
        params = self.weights
        for _ in range(epochs):
            for batch, labels in train_loader:
                params = opt.step(lambda w: self.loss(w, batch, labels), params)
        self.weights = params

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = []
        for x in X:
            out = self.circuit(x, self.weights)
            preds.append(out[0] > 0)
        return np.array(preds).astype(int)
