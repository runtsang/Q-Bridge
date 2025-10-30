"""Quantum fraud detection model using a variational circuit and measurement‑based post‑processing."""

from __future__ import annotations

import pennylane as qml
from pennylane import qnn
import pennylane.numpy as np
from dataclasses import dataclass
from typing import Iterable, Sequence

__all__ = ["FraudLayerParameters", "FraudDetectionEnhanced"]


@dataclass
class FraudLayerParameters:
    """Parameters for a single photonic layer, repurposed as variational angles."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


class FraudDetectionEnhanced:
    """Quantum fraud detection model with a variational ansatz and classical post‑processing."""
    def __init__(self, dev: qml.Device, input_params: FraudLayerParameters,
                 layers: Iterable[FraudLayerParameters], depth: int = 2):
        self.dev = dev
        self.input_params = input_params
        self.layers = layers
        self.depth = depth

        # Initialize trainable parameters for the ansatz
        self.trainable_params = np.array([0.0] * (depth * 8), dtype=np.float64)

        # Simple classical post‑processing network
        self.classifier = qnn.MLClassifier(
            output_dim=1,
            hidden_layers=[8],
            device=self.dev,
            seed=42,
        )

        # Build the QNode
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")

    def _circuit(self, *params) -> np.ndarray:
        """Variational circuit applying layers of rotations and CNOTs."""
        for d in range(self.depth):
            idx = d * 8
            theta = params[idx]
            phi = params[idx + 1]
            alpha = params[idx + 2]
            beta = params[idx + 3]
            gamma = params[idx + 4]
            delta = params[idx + 5]
            eps = params[idx + 6]
            zeta = params[idx + 7]

            qml.RY(theta, wires=0)
            qml.RZ(phi, wires=1)
            qml.RX(alpha, wires=0)
            qml.RY(beta, wires=1)
            qml.CNOT(wires=[0, 1])

            qml.RZ(gamma, wires=0)
            qml.RX(delta, wires=1)
            qml.CNOT(wires=[1, 0])

            qml.RY(eps, wires=0)
            qml.RZ(zeta, wires=1)

        # Return measurement probabilities for both qubits
        return qml.probs(wires=[0, 1])

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: encode the input, run the variational circuit, then classify."""
        # Simple angle encoding of the two features
        theta_in = x[0] * np.pi
        phi_in = x[1] * np.pi
        qml.RY(theta_in, wires=0)
        qml.RZ(phi_in, wires=1)

        # Execute the circuit with current trainable parameters
        probs = self.qnode(*self.trainable_params)
        # Classify the measurement probabilities
        return self.classifier(probs)

    @classmethod
    def build_fraud_detection_program(cls, input_params: FraudLayerParameters,
                                      layers: Iterable[FraudLayerParameters],
                                      depth: int = 2) -> "FraudDetectionEnhanced":
        """Convenience constructor mirroring the original API."""
        dev = qml.device("default.qubit", wires=2)
        return cls(dev, input_params, layers, depth)
