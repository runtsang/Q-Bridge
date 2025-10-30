"""
EstimatorQNN: A 2‑qubit variational quantum regression model built with Pennylane.

The circuit consists of parameterized rotations on each qubit followed by a
CNOT entangling gate.  The model estimates the expectation value of a
Pauli‑Z observable, which serves as the regression output.  A simple
gradient‑based optimizer is exposed via the ``train`` method.
"""

from __future__ import annotations

import pennylane as qml
import numpy as np
from typing import Iterable, Tuple

__all__ = ["EstimatorQNN"]


def EstimatorQNN(
    n_qubits: int = 2,
    n_layers: int = 2,
    device_name: str = "default.qubit",
    seed: int | None = None,
) -> "EstimatorQNNModel":
    """
    Construct a variational quantum regression model.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit (default 2).
    n_layers : int
        Number of repeatable rotation–entanglement blocks.
    device_name : str
        Pennylane device to use for simulation.
    seed : int | None
        Random seed for weight initialization.

    Returns
    -------
    EstimatorQNNModel
        Object exposing ``predict`` and ``train`` methods.
    """
    dev = qml.device(device_name, wires=n_qubits)

    # Parameterised circuit
    def circuit(inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        for i in range(n_qubits):
            qml.RX(inputs[i], wires=i)
            qml.RY(weights[i], wires=i)

        for _ in range(n_layers):
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            for i in range(n_qubits):
                qml.RX(weights[i], wires=i)
                qml.RY(weights[i], wires=i)

        return qml.expval(qml.PauliZ(0))

    qnode = qml.QNode(circuit, dev, interface="autograd")

    class EstimatorQNNModel:
        def __init__(self) -> None:
            # Initialise weights uniformly in [-π, π]
            rng = np.random.default_rng(seed)
            self.weights = rng.uniform(-np.pi, np.pi, size=n_qubits)

        def predict(self, X: Iterable[np.ndarray]) -> np.ndarray:
            """
            Compute the regression output for a batch of inputs.

            Parameters
            ----------
            X : Iterable[ndarray]
                Each element must be a 1‑D array of length ``n_qubits``.

            Returns
            -------
            ndarray
                Prediction values.
            """
            preds = [qnode(x, self.weights) for x in X]
            return np.array(preds)

        def train(
            self,
            X: Iterable[np.ndarray],
            y: Iterable[float],
            lr: float = 0.01,
            epochs: int = 200,
            loss_fn: str = "mse",
        ) -> Tuple[np.ndarray, np.ndarray]:
            """
            Train the variational circuit using a simple gradient descent loop.

            Parameters
            ----------
            X : Iterable[ndarray]
                Training inputs.
            y : Iterable[float]
                Target values.
            lr : float
                Learning rate.
            epochs : int
                Number of epochs.
            loss_fn : str
                Loss function name; currently supports'mse'.

            Returns
            -------
            tuple
                (loss_history, weight_history)
            """
            opt = qml.GradientDescentOptimizer(lr)
            loss_history = []
            weight_history = []

            for _ in range(epochs):
                def loss_fn_model(w):
                    preds = [qnode(x, w) for x in X]
                    preds = np.array(preds)
                    if loss_fn == "mse":
                        return np.mean((preds - y) ** 2)
                    raise ValueError(f"Unsupported loss: {loss_fn}")

                self.weights = opt.step(loss_fn_model, self.weights)
                loss_history.append(loss_fn_model(self.weights))
                weight_history.append(self.weights.copy())

            return np.array(loss_history), np.array(weight_history)

    return EstimatorQNNModel()
