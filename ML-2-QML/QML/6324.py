"""Quantum variational regressor that mirrors the classical EstimatorQNN.

Implemented with Pennylane, the model exposes the same factory function
EstimatorQNN() and supports a configurable number of qubits and variational
layers.  Gradients are obtained via the built‑in autograd interface.
"""

import pennylane as qml
import numpy as np
from pennylane import numpy as pnp
from typing import Sequence


class EstimatorQNN:
    """Variational quantum circuit for regression.

    Parameters
    ----------
    n_qubits: int
        Number of qubits in the circuit.
    layers: int
        Number of variational layers.
    data_map: callable | None
        Function that maps a 2‑dimensional input to rotation angles.
    """

    def __init__(self, n_qubits: int = 2, layers: int = 2,
                 data_map: callable | None = None) -> None:
        self.n_qubits = n_qubits
        self.layers = layers
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.data_map = data_map or (lambda x: x)

        # Trainable parameters: one rotation per qubit per layer
        self.weights = pnp.random.randn(layers, n_qubits, 3, requires_grad=True)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs: Sequence[float], weights: np.ndarray) -> float:
            # Data embedding
            for i, w in enumerate(inputs):
                qml.RY(w, wires=i)
            # Variational layers
            for layer in range(self.layers):
                for q in range(self.n_qubits):
                    qml.RX(weights[layer, q, 0], wires=q)
                    qml.RY(weights[layer, q, 1], wires=q)
                    qml.RZ(weights[layer, q, 2], wires=q)
                # Entangling layer
                for q in range(self.n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
            # Output observable
            return qml.expval(qml.PauliZ(0))
        self._circuit = circuit

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """Predict outputs for a batch of inputs.

        Parameters
        ----------
        inputs: array_like, shape (n_samples, 2)
            Batch of 2‑dimensional data points.
        """
        preds = []
        for x in inputs:
            preds.append(self._circuit(self.data_map(x), self.weights))
        return np.array(preds)

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """Mean squared error loss."""
        preds = self.__call__(x)
        return np.mean((preds - y) ** 2)

    def gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return gradients w.r.t. the trainable weights."""
        return qml.grad(self.loss)(x, y)


def EstimatorQNN(**kwargs) -> EstimatorQNN:
    """Factory that creates a quantum EstimatorQNN instance.

    All keyword arguments are forwarded to EstimatorQNN.__init__.
    """
    return EstimatorQNN(**kwargs)


__all__ = ["EstimatorQNN"]
