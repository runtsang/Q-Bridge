"""Quantum version of the fraud detection model using PennyLane."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pennylane as qml

@dataclass
class FraudLayerParams:
    """Parameters for a single photonic layer, reused in the quantum model."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

class FraudDetectionModel:
    """PennyLane based variational circuit that emulates the photonic fraud detection.

    The circuit accepts a 2‑dimensional input vector and applies a sequence of
    parameterised gates that mirror the structure of the classical model.
    """
    def __init__(
        self,
        input_params: FraudLayerParams,
        layers: Iterable[FraudLayerParams],
        device: str = "default.qubit",
        shots: int = 1024,
    ) -> None:
        self.device = qml.device(device, wires=2, shots=shots)
        self.input_params = input_params
        self.layers = list(layers)
        self.qnode = qml.QNode(self._circuit, self.device)

    def _apply_layer(self, params: FraudLayerParams, clip: bool) -> None:
        """Encode a single layer into the quantum circuit."""
        # Beam‑splitter like mixing
        qml.RZ(params.bs_theta, wires=0)
        qml.RY(params.bs_phi, wires=1)
        # Phase shifts
        for i, phase in enumerate(params.phases):
            qml.RZ(phase, wires=i)
        # Squeezing and displacement mapped to rotations
        for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
            r_clipped = np.clip(r, -5, 5) if clip else r
            qml.Rot(r_clipped, phi, 0, wires=i)
        for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
            r_clipped = np.clip(r, -5, 5) if clip else r
            qml.Rot(r_clipped, phi, 0, wires=i)
        # Kerr non‑linearity approximated by a controlled‑phase
        for i, k in enumerate(params.kerr):
            k_clipped = np.clip(k, -1, 1) if clip else k
            qml.CZ(wires=[i, (i + 1) % 2])  # placeholder for Kerr effect
            qml.RZ(k_clipped, wires=i)

    def _circuit(self, x: np.ndarray) -> np.ndarray:
        """Variational circuit that produces a real‑valued output."""
        qml.QubitStateVector(x, wires=[0, 1])
        self._apply_layer(self.input_params, clip=False)
        for layer in self.layers:
            self._apply_layer(layer, clip=True)
        return qml.expval(qml.PauliZ(0))

    def __call__(self, x: np.ndarray) -> float:
        """Forward pass through the quantum circuit."""
        return float(self.qnode(x))

    def train(
        self,
        data: Sequence[np.ndarray],
        labels: Sequence[float],
        lr: float = 0.01,
        epochs: int = 100,
    ) -> None:
        """Simple stochastic gradient descent training loop."""
        opt = qml.GradientDescentOptimizer(lr)
        for _ in range(epochs):
            for x, y in zip(data, labels):
                loss, grads = qml.gradients.param_shift(self.qnode, argnum=0)(x, y)
                opt.step(grads)

__all__ = ["FraudLayerParams", "FraudDetectionModel"]
