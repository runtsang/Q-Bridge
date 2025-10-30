"""Hybrid quantum‑classical regressor using PennyLane and parameter‑shift gradients."""
from __future__ import annotations

import pennylane as qml
from pennylane import numpy as np
from pennylane import qnodes
from pennylane.optimize import AdamOptimizer

__all__ = ["EstimatorQNN"]


class EstimatorQNN:
    """
    A hybrid model that couples a PennyLane variational circuit to a classical
    linear head. The circuit outputs a single expectation value of a Pauli‑Z
    observable, which is then passed through a learnable bias.

    Architecture:
        - 2‑qubit variational circuit with Ry‑RZ entangling layers
        - Expectation of Pauli‑Z on qubit 0
        - Linear layer: y = w * E + b
    """

    def __init__(self, n_layers: int = 3, seed: int | None = None) -> None:
        self.n_qubits = 2
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        # Initialize variational parameters
        self.params = np.random.randn(self.n_layers, self.n_qubits, 3) * 0.1
        # Classical linear head parameters
        self.w = np.array([1.0], requires_grad=True)
        self.b = np.array([0.0], requires_grad=True)

    def circuit(self, x: float, params: np.ndarray) -> float:
        """Single‑shot circuit returning <Z> on qubit 0."""
        # Encode data (simple rotation)
        qml.RY(x, wires=0)

        # Variational layers
        for layer in params:
            for qubit in range(self.n_qubits):
                qml.RY(layer[qubit, 0], wires=qubit)
                qml.RZ(layer[qubit, 1], wires=qubit)
            # Entanglement
            qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0))

    qnode = qml.QNode(circuit, dev, interface="autograd")

    def __call__(self, x: float) -> float:
        """Forward pass returning a scalar prediction."""
        expval = self.qnode(x, self.params)
        return float(self.w * expval + self.b)

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------
    def loss(self, pred: float, target: float) -> float:
        """Mean‑squared error."""
        return (pred - target) ** 2

    def train_step(
        self,
        data: float,
        target: float,
        opt: AdamOptimizer,
        lr: float = 0.1,
    ) -> float:
        """Single training step returning loss."""
        def loss_fn(params, w, b):
            expval = self.qnode(data, params)
            pred = w * expval + b
            return (pred - target) ** 2

        grads = qml.grad(loss_fn)(self.params, self.w, self.b)
        self.params -= lr * grads[0]
        self.w -= lr * grads[1]
        self.b -= lr * grads[2]
        return float(loss_fn(self.params, self.w, self.b))
