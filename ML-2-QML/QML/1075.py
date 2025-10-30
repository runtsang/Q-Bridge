"""Hybrid quantum‑classical regressor using PennyLane variational circuits."""

import pennylane as qml
import numpy as np
from pennylane.optimize import AdamOptimizer
from typing import Callable

class EstimatorQNN:
    """A variational quantum circuit with classical post‑processing for regression."""
    def __init__(
        self,
        n_qubits: int = 2,
        n_layers: int = 2,
        n_params: int | None = None,
        observable: str | None = None,
        device: str | None = None,
    ) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.device = device or qml.device("default.qubit", wires=n_qubits)
        self.n_params = n_params if n_params is not None else n_qubits * n_layers * 3
        self.observable = observable or ("Z" * n_qubits)

        @qml.qnode(self.device)
        def circuit(inputs, weights):
            # Encode inputs via RY rotations
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)
            # Variational layers
            idx = 0
            for _ in range(n_layers):
                for i in range(n_qubits):
                    qml.Rot(
                        weights[idx], weights[idx + 1], weights[idx + 2], wires=i
                    )
                    idx += 3
                # Entangling layer
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            # Measurement
            return qml.expval(qml.PauliZ(wires=0))

        self.circuit = circuit
        self.weights = 0.01 * np.random.randn(self.n_params)

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """Evaluate circuit for a single input vector."""
        return self.circuit(inputs, self.weights)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 200,
        lr: float = 0.01,
        verbose: bool = False,
    ) -> None:
        """Train the variational parameters via Adam."""
        opt = AdamOptimizer(lr)
        loss_fn = lambda w: np.mean((self.circuit(X, w) - y) ** 2)
        for epoch in range(epochs):
            self.weights = opt.step(loss_fn, self.weights)
            if verbose and (epoch + 1) % 20 == 0:
                loss = loss_fn(self.weights)
                print(f"Epoch {epoch+1}/{epochs} – loss: {loss:.6f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions for a batch of inputs."""
        return np.array([self.circuit(x, self.weights) for x in X])

__all__ = ["EstimatorQNN"]
