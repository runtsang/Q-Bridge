"""Variational regression estimator built with Pennylane."""

import pennylane as qml
import numpy as np
from pennylane import numpy as pnp
from typing import Callable, Sequence


class EstimatorQNN:
    """Quantum variational regressor using a multi‑layer entangled circuit."""

    def __init__(
        self,
        num_qubits: int = 4,
        layers: int = 3,
        cost_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
    ) -> None:
        self.num_qubits = num_qubits
        self.layers = layers
        self.dev = qml.device("default.qubit", wires=num_qubits)
        self.params = pnp.random.uniform(0, 2 * np.pi, size=(layers, num_qubits))
        self.cost_fn = cost_fn or self.mse

        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs: Sequence[float], weights: np.ndarray) -> np.ndarray:
            # Data encoding
            for i, x in enumerate(inputs):
                qml.RX(x, wires=i)
            # Parameterised layers
            for l in range(layers):
                for q in range(num_qubits):
                    qml.RY(weights[l, q], wires=q)
                # Entangling block
                for q in range(num_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
                qml.CNOT(wires=[num_qubits - 1, 0])
            # Measurement
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Return the expectation value for each input sample."""
        return np.array([self.circuit(x, self.params) for x in inputs])

    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 200,
        lr: float = 0.01,
    ) -> list[float]:
        """Gradient‑based training of the variational parameters."""
        opt = qml.GradientDescentOptimizer(stepsize=lr)
        loss_hist: list[float] = []

        for _ in range(epochs):
            loss = self.cost_fn(self.predict(x_train), y_train)
            loss_hist.append(loss)
            grads = qml.grad(self.circuit)(x_train, self.params)
            self.params = opt.step(grads, self.params)
        return loss_hist

    @staticmethod
    def mse(preds: np.ndarray, targets: np.ndarray) -> float:
        return np.mean((preds - targets) ** 2)


__all__ = ["EstimatorQNN"]
