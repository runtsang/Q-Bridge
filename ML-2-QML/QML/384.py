"""Quantum classifier using Pennylane variational circuit.

Provides a quantum circuit, training loop with parameter‑shift, and a lightweight
linear post‑processor that matches the classical interface.
"""

from __future__ import annotations

import pennylane as qml
import numpy as np
from typing import List, Tuple


class QuantumClassifierModel:
    """
    Variational quantum classifier that mirrors the classical interface.
    """

    def __init__(
        self,
        num_qubits: int,
        depth: int,
        obs: List[qml.operation.Operator] | None = None,
        device: str | None = None,
    ) -> None:
        """
        Parameters
        ----------
        num_qubits
            Number of qubits (features).
        depth
            Number of ansatz layers.
        obs
            List of Pauli observables measured per qubit. If None, Z on each qubit.
        device
            Pennylane device name. Defaults to "default.qubit".
        """
        self.num_qubits = num_qubits
        self.depth = depth
        self.obs = obs if obs is not None else [qml.PauliZ(i) for i in range(num_qubits)]
        self.dev = qml.device(device or "default.qubit", wires=num_qubits)

        # Parameter vector for data encoding and ansatz
        self.enc_params = np.random.uniform(0, 2 * np.pi, num_qubits)
        self.ansatz_params = np.random.uniform(0, 2 * np.pi, num_qubits * depth)

        # Define the circuit
        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs: np.ndarray, enc_params: np.ndarray, ansatz_params: np.ndarray):
            # Data encoding
            for idx, wire in enumerate(range(num_qubits)):
                qml.RX(inputs[idx] * enc_params[wire], wires=wire)
            # Ansatz
            ptr = 0
            for _ in range(depth):
                for wire in range(num_qubits):
                    qml.RY(ansatz_params[ptr], wires=wire)
                    ptr += 1
                for wire in range(num_qubits - 1):
                    qml.CZ(wires=[wire, wire + 1])
            # Measurements
            return [qml.expval(obs) for obs in self.obs]

        self.circuit = circuit

        # Classical post‑processing: simple linear classifier
        self.W = np.random.randn(len(self.obs), 2) * 0.1
        self.b = np.zeros(2)

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass returning logits.
        """
        out = self.circuit(inputs, self.enc_params, self.ansatz_params)
        logits = out @ self.W + self.b
        return logits

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Cross‑entropy loss over a batch.
        """
        logits = self.__call__(X)
        probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        ce = -np.log(probs[range(len(y)), y]).mean()
        return ce

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        epochs: int = 50,
        lr: float = 0.01,
        batch_size: int = 16,
    ) -> None:
        """
        Train using Adam on the combined quantum‑classical parameters.
        """
        opt = qml.AdamOptimizer(stepsize=lr)

        for epoch in range(epochs):
            idx = np.arange(len(X))
            np.random.shuffle(idx)
            for start in range(0, len(X), batch_size):
                batch_idx = idx[start : start + batch_size]
                Xb, yb = X[batch_idx], y[batch_idx]
                params = np.concatenate(
                    [self.enc_params, self.ansatz_params, self.W.ravel(), self.b]
                )
                params_grad = opt.gradient(lambda p: self.loss(Xb, yb), params)
                params = opt.step(lambda p: self.loss(Xb, yb), params, params_grad)

                # unpack
                enc_len = self.enc_params.size
                ans_len = self.ansatz_params.size
                self.enc_params = params[:enc_len]
                self.ansatz_params = params[enc_len : enc_len + ans_len]
                w_start = enc_len + ans_len
                w_end = w_start + self.W.size
                self.W = params[w_start:w_end].reshape(self.W.shape)
                self.b = params[w_end:].reshape(self.b.shape)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return class indices.
        """
        logits = self.__call__(X)
        return np.argmax(logits, axis=1)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return accuracy.
        """
        preds = self.predict(X)
        return (preds == y).mean()


__all__ = ["QuantumClassifierModel"]
