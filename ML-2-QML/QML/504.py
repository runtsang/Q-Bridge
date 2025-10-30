"""Hybrid quantum-classical fraud detection circuit using PennyLane."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, List

import pennylane as qml
import numpy as np

@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic layer."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

    def clip(self, bound: float = 5.0) -> "FraudLayerParameters":
        return FraudLayerParameters(
            bs_theta=np.clip(self.bs_theta, -bound, bound),
            bs_phi=np.clip(self.bs_phi, -bound, bound),
            phases=tuple(np.clip(np.array(self.phases), -bound, bound)),
            squeeze_r=tuple(np.clip(np.array(self.squeeze_r), -bound, bound)),
            squeeze_phi=tuple(np.clip(np.array(self.squeeze_phi), -bound, bound)),
            displacement_r=tuple(np.clip(np.array(self.displacement_r), -bound, bound)),
            displacement_phi=tuple(np.clip(np.array(self.displacement_phi), -bound, bound)),
            kerr=tuple(np.clip(np.array(self.kerr), -bound, bound)),
        )

class FraudDetection:
    """Quantum photonic fraud detection circuit with variational training."""
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
    ) -> None:
        self.input_params = input_params
        self.layers = list(layers)
        self.wires = 2
        self.device = qml.device("default.qubit", wires=self.wires)
        self.params = self._flatten_params()

    def _flatten_params(self) -> np.ndarray:
        """Flatten all layer parameters into a single vector."""
        vec: List[float] = []
        for layer in [self.input_params] + self.layers:
            vec.extend([layer.bs_theta, layer.bs_phi])
            vec.extend(layer.phases)
            vec.extend(layer.squeeze_r)
            vec.extend(layer.squeeze_phi)
            vec.extend(layer.displacement_r)
            vec.extend(layer.displacement_phi)
            vec.extend(layer.kerr)
        return np.array(vec, dtype=np.float32)

    def _circuit(self, x: Tuple[float, float], params: np.ndarray) -> float:
        """Variational circuit returning the Pauli‑Z⊗Z expectation."""
        @qml.qnode(self.device, interface="autograd")
        def circuit():
            # Encode classical data
            qml.RX(x[0], wires=0)
            qml.RY(x[1], wires=1)

            idx = 0
            # Input layer
            qml.RZ(params[idx], wires=0); idx += 1
            qml.RZ(params[idx], wires=1); idx += 1

            # Hidden layers
            for _ in self.layers:
                qml.RZ(params[idx], wires=0); idx += 1
                qml.RZ(params[idx], wires=1); idx += 1
                qml.CNOT(wires=[0, 1])

            # Measurement
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        return circuit()

    def loss(self, X: np.ndarray, y: np.ndarray, params: np.ndarray) -> float:
        """Binary cross‑entropy loss over a batch."""
        preds = np.array([self._circuit(tuple(x), params) for x in X])
        preds = (preds + 1) / 2  # map from [-1,1] to [0,1]
        loss = -np.mean(
            y * np.log(preds + 1e-9) + (1 - y) * np.log(1 - preds + 1e-9)
        )
        return loss

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        lr: float = 0.01,
        batch_size: int = 32,
    ) -> List[float]:
        """Stochastic gradient descent loop using parameter shift."""
        opt = qml.GradientDescentOptimizer(lr)
        losses: List[float] = []

        for epoch in range(epochs):
            idx = np.random.permutation(len(X))
            X_shuffled, y_shuffled = X[idx], y[idx]
            for start in range(0, len(X), batch_size):
                end = start + batch_size
                batch_X, batch_y = X_shuffled[start:end], y_shuffled[start:end]
                # Compute gradient w.r.t. current parameters
                grads = opt.gradient(
                    lambda p: self.loss(batch_X, batch_y, p), self.params
                )
                self.params = opt.step(grads, self.params)

            epoch_loss = self.loss(X, y, self.params)
            losses.append(epoch_loss)
            if epoch % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch} loss {epoch_loss:.4f}")

        return losses

    def predict(
        self,
        X: np.ndarray,
        threshold: float = 0.5,
    ) -> np.ndarray:
        preds = np.array([self._circuit(tuple(x), self.params) for x in X])
        preds = (preds + 1) / 2
        return (preds >= threshold).astype(int)

__all__ = ["FraudLayerParameters", "FraudDetection"]
