"""Hybrid estimator that couples a classical residual network with a variational quantum circuit."""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.qnn import CircuitQNN
from pennylane.optimize import Adam
from pennylane import qchem
from pennylane import device as qml_device
from typing import Callable

# Classical residual network (same as the ML version)
class EstimatorQNN:
    def __init__(self,
                 input_dim: int = 2,
                 hidden_dim: int = 8,
                 output_dim: int = 1,
                 use_dropout: bool = False,
                 dropout_prob: float = 0.1) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_dropout = use_dropout
        self.dropout_prob = dropout_prob
        self.params = pnp.random.uniform(-np.pi, np.pi,
                                         (input_dim + hidden_dim + hidden_dim,))

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """Classical forward pass."""
        # Simple linear layers with a tanh nonlinearity
        h = pnp.tanh(pnp.dot(inputs, self.params[:self.input_dim]) +
                     self.params[self.input_dim:self.input_dim + self.hidden_dim])
        h = pnp.tanh(pnp.dot(h, self.params[self.input_dim + self.hidden_dim:]) +
                     self.params[self.input_dim + self.hidden_dim + self.hidden_dim:])
        return h

# Variational quantum circuit
def _create_vqc(num_qubits: int, depth: int = 2) -> qml.QNode:
    @qml.qnode(qml_device('default.qubit', wires=num_qubits))
    def circuit(inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Variational circuit with parameter‑shift compatible gates."""
        for i in range(num_qubits):
            qml.RX(inputs[i], wires=i)
            qml.RZ(weights[i], wires=i)
        for _ in range(depth):
            for i in range(num_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            for i in range(num_qubits):
                qml.RY(weights[num_qubits + i], wires=i)
        return qml.expval(qml.PauliZ(0))
    return circuit

# Hybrid estimator combining classical and quantum parts
class HybridEstimatorQNN:
    def __init__(self,
                 num_qubits: int = 1,
                 vqc_depth: int = 2,
                 learning_rate: float = 1e-3,
                 epochs: int = 200):
        self.num_qubits = num_qubits
        self.vqc = _create_vqc(num_qubits, vqc_depth)
        self.classical_net = EstimatorQNN()
        self.opt = Adam(learning_rate)
        self.epochs = epochs
        self.loss_history = []

    def loss(self, params: np.ndarray, inputs: np.ndarray, targets: np.ndarray) -> float:
        """Mean‑squared error between quantum output and target."""
        preds = self.vqc(inputs, params)
        return np.mean((preds - targets) ** 2)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the hybrid model using the parameter‑shift rule."""
        params = pnp.random.uniform(-np.pi, np.pi, self.vqc.num_params)
        for epoch in range(self.epochs):
            self.opt.step(lambda p: self.loss(p, X, y), params)
            self.loss_history.append(self.loss(params, X, y))
        self.params = params

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions from the quantum circuit."""
        return self.vqc(X, self.params)

__all__ = ["HybridEstimatorQNN"]
