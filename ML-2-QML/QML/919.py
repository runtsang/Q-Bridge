"""QCNNGen199 – Quantum convolutional neural network built with Pennylane.

The quantum implementation replaces the hard‑coded Qiskit circuits with a
Pennylane variational QCNN that is fully differentiable via the parameter‑shift
rule.  It includes a feature map, convolutional and pooling layers, and a
single‑qubit measurement that yields a scalar output suitable for binary
classification.
"""

from __future__ import annotations

import pennylane as qml
import numpy as np
from pennylane import numpy as pnp
from pennylane.optimize import AdamOptimizer
from typing import Callable, Iterable, Tuple


class QCNNGen199:
    """Hybrid QCNN model using Pennylane.

    Parameters
    ----------
    n_qubits : int
        Total number of qubits in the circuit (default 8).
    n_layers : int
        Number of convolution–pooling stages (default 3).
    """

    def __init__(self, n_qubits: int = 8, n_layers: int = 3) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self._build_circuit()

    def _conv_layer(self, wires: Tuple[int,...], params: Iterable[float]) -> None:
        """Two‑qubit convolution block."""
        for i, (w1, w2) in enumerate(zip(wires[0::2], wires[1::2])):
            qml.RZ(params[3 * i], wires=w1)
            qml.CNOT(w1, w2)
            qml.RY(params[3 * i + 1], wires=w2)
            qml.CNOT(w2, w1)
            qml.RY(params[3 * i + 2], wires=w2)

    def _pool_layer(self, wires: Tuple[int,...], params: Iterable[float]) -> None:
        """Two‑qubit pooling block."""
        for i, (w1, w2) in enumerate(zip(wires[0::2], wires[1::2])):
            qml.RZ(params[3 * i], wires=w1)
            qml.CNOT(w1, w2)
            qml.RY(params[3 * i + 1], wires=w2)
            qml.CNOT(w2, w1)

    def _feature_map(self, x: np.ndarray) -> None:
        """Z‑feature map acting on all qubits."""
        for i, val in enumerate(x):
            qml.RZ(val, wires=i)

    def _build_circuit(self) -> None:
        """Construct the variational circuit and cost function."""
        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs: np.ndarray, weights: np.ndarray) -> float:
            # Feature embedding
            self._feature_map(inputs)

            # Variational layers
            idx = 0
            for layer in range(self.n_layers):
                # Convolution
                conv_params = weights[idx : idx + 3 * (self.n_qubits // (2 ** layer))]
                self._conv_layer(
                    tuple(range(2 ** layer, 2 ** (layer + 1))),
                    conv_params,
                )
                idx += len(conv_params)

                # Pooling
                pool_params = weights[idx : idx + 3 * (self.n_qubits // (2 ** (layer + 1)))]
                self._pool_layer(
                    tuple(range(2 ** (layer + 1), 2 ** (layer + 2))),
                    pool_params,
                )
                idx += len(pool_params)

            # Measurement
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

        # Initialise weights
        self.weights = pnp.random.random(self.circuit.num_params)

    def __call__(self, inputs: np.ndarray) -> float:
        """Evaluate the QCNN on a single input."""
        return float(self.circuit(inputs, self.weights))

    def loss(self, inputs: np.ndarray, target: float) -> float:
        """Binary cross‑entropy loss."""
        pred = self.__call__(inputs)
        return -(target * np.log(pred + 1e-10) + (1 - target) * np.log(1 - pred + 1e-10))

    def train_step(self, inputs: np.ndarray, target: float, lr: float = 0.01) -> None:
        """Single optimisation step using Adam."""
        opt = AdamOptimizer(lr)
        self.weights, _ = opt.step_and_cost(
            lambda w: self.loss(inputs, target), self.weights
        )

    def predict(self, dataset: Iterable[np.ndarray]) -> np.ndarray:
        """Vectorised prediction over a dataset."""
        return np.array([self.__call__(x) for x in dataset])

    def fit(
        self,
        X: Iterable[np.ndarray],
        y: Iterable[float],
        epochs: int = 100,
        lr: float = 0.01,
    ) -> None:
        """Simple training loop over the entire dataset."""
        for epoch in range(epochs):
            for x, t in zip(X, y):
                self.train_step(x, t, lr)
            if epoch % 10 == 0:
                loss = np.mean([self.loss(x, t) for x, t in zip(X, y)])
                print(f"Epoch {epoch} – loss: {loss:.4f}")

__all__ = ["QCNNGen199"]
