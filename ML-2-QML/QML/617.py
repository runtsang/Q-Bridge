"""Hybrid quantum‑classical QCNN implemented with Pennylane."""

from __future__ import annotations

import pennylane as qml
import pennylane.numpy as np
from pennylane import numpy as pnp
from typing import Sequence


class QCNNModel:
    """A hybrid QCNN: a feature‑map, several conv‑pool layers and a classical read‑out."""

    def __init__(
        self,
        num_qubits: int = 8,
        conv_layers: int = 3,
        pool_layers: int = 3,
        depth_per_layer: int = 2,
        device: str | qml.Device | None = None,
        seed: int | None = 42,
    ) -> None:
        self.num_qubits = num_qubits
        self.conv_layers = conv_layers
        self.pool_layers = pool_layers
        self.depth_per_layer = depth_per_layer
        self.device = device or qml.device("default.qubit", wires=num_qubits)
        self.seed = seed
        self._build_ansatz()

    def _convolution(self, qubits: Sequence[int], params: Sequence[float]) -> None:
        """Two‑qubit convolution block: a parametrised entangling pattern."""
        for idx, q in enumerate(qubits):
            qml.RZ(params[idx], wires=q)
            qml.RY(params[idx + len(qubits)], wires=q)
            if idx < len(qubits) - 1:
                qml.CNOT(wires=[q, qubits[idx + 1]])

    def _pooling(self, qubits: Sequence[int], params: Sequence[float]) -> None:
        """Pooling: measure and reset a qubit, then entangle the remaining ones."""
        for idx, q in enumerate(qubits):
            qml.RZ(params[idx], wires=q)
            qml.CNOT(wires=[q, (q + 1) % self.num_qubits])

    def _build_ansatz(self) -> None:
        """Construct a parametric circuit with alternating conv & pool layers."""
        self.params = {}
        for l in range(self.conv_layers + self.pool_layers):
            layer_name = f"layer_{l}"
            num_params = self.num_qubits * self.depth_per_layer
            self.params[layer_name] = np.random.uniform(0, 2 * np.pi, num_params)

    def circuit(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: embed classical data, run the ansatz, and read out."""
        dev = self.device
        dev.reset()

        # Feature mapping: simple angle embedding
        for i, val in enumerate(x):
            qml.RY(val, wires=i)

        # Apply conv & pool layers
        param_idx = 0
        for l in range(self.conv_layers + self.pool_layers):
            layer_name = f"layer_{l}"
            layer_params = self.params[layer_name]
            if l % 2 == 0:  # convolution
                self._convolution(range(self.num_qubits), layer_params)
            else:  # pooling
                self._pooling(range(self.num_qubits), layer_params)

        # Classical read‑out: expectation of Z on first qubit
        return qml.expval(qml.PauliZ(0))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Batch inference."""
        return np.array([self.circuit(x) for x in X])

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 50,
        lr: float = 0.01,
        loss_fn: str = "binary_crossentropy",
    ) -> None:
        """Simple training loop using Pennylane's autograd."""
        opt = qml.AdamOptimizer(stepsize=lr)
        for epoch in range(epochs):
            def loss_fn_wrapper(params):
                loss = 0.0
                for x, y in zip(X_train, y_train):
                    pred = qml.apply(
                        lambda: self.circuit(x),
                        weight_params=params,
                    )
                    if loss_fn == "binary_crossentropy":
                        loss += -y * np.log(pred + 1e-8) - (1 - y) * np.log(1 - pred + 1e-8)
                return loss / len(X_train)

            self.params = opt.step(loss_fn_wrapper, self.params)

    def __repr__(self) -> str:
        return f"<QCNNModel layers={self.conv_layers + self.pool_layers} qubits={self.num_qubits}>"

__all__ = ["QCNNModel"]
