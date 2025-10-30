"""Variational quantum fraud detection model using PennyLane."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pennylane as qml
import numpy as np
import torch


@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic layer (kept for compatibility)."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


class FraudDetectionModel:
    """Variational quantum fraud detection model using PennyLane."""
    def __init__(self, input_params: FraudLayerParameters, layers: Iterable[FraudLayerParameters]) -> None:
        self.input_params = input_params
        self.layers = list(layers)
        self.dev = qml.device("default.qubit", wires=2)
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")

    def _circuit(self, x: np.ndarray, params: np.ndarray) -> float:
        # Data re-upload: encode the two input features as rotations
        qml.RY(x[0], wires=0)
        qml.RY(x[1], wires=1)

        # Apply a variational layer for each FraudLayerParameters
        # The params array is split into blocks of four angles per layer
        for i, layer_params in enumerate(self.layers):
            # Map the 4 variational angles to the circuit
            theta = params[i * 4 + 0]
            phi = params[i * 4 + 1]
            alpha = params[i * 4 + 2]
            beta = params[i * 4 + 3]

            # Data-dependent gates
            qml.RY(theta, wires=0)
            qml.RZ(phi, wires=1)

            # Entanglement pattern
            qml.CNOT(wires=[0, 1])

            # Additional rotations
            qml.RY(alpha, wires=1)
            qml.RZ(beta, wires=0)

        # Measurement of PauliZ on the first qubit
        return qml.expval(qml.PauliZ(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning the expectation value as a probability."""
        x_np = x.detach().cpu().numpy()
        # Initialize variational parameters randomly
        num_params = len(self.layers) * 4
        params = np.random.randn(num_params)
        return self.qnode(x_np, params)

    @staticmethod
    def build_fraud_detection_program(
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
    ) -> "FraudDetectionModel":
        """Convenience constructor mirroring the original seed API."""
        return FraudDetectionModel(input_params, layers)


__all__ = ["FraudLayerParameters", "FraudDetectionModel"]
