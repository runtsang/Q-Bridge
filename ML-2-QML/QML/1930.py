"""Pennylane implementation of the photonic fraud‑detection circuit.

The module maps the original Strawberry‑Fields construction to a qubit‑based variational
circuit and exposes a `build_fraud_detection_program` function that returns a QNode.
"""

from __future__ import annotations

import pennylane as qml
from pennylane import numpy as np
from dataclasses import dataclass
from typing import Iterable, List

@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]
    shift: float = 0.0

class FraudDetection:
    """Quantum fraud‑detection circuit built from a list of layer parameters."""
    def __init__(self, input_params: FraudLayerParameters, hidden_params: Iterable[FraudLayerParameters], device=None):
        self.device = device or qml.device("default.qubit", wires=2)
        self.input_params = input_params
        self.hidden_params = list(hidden_params)
        self.qnode = qml.qnode(self._circuit, device=self.device)

    def _circuit(self) -> float:
        """Variational circuit that mimics the photonic layer structure."""
        # Input layer
        self._apply_layer(self.input_params)
        # Hidden layers
        for p in self.hidden_params:
            self._apply_layer(p)
        # Measurement
        return qml.expval(qml.PauliZ(0))

    def _apply_layer(self, p: FraudLayerParameters) -> None:
        """Encode a single FraudLayerParameters into the qubit circuit."""
        # Beam‑splitter analogue: use rotations
        qml.RY(p.bs_theta, wires=0)
        qml.RZ(p.bs_phi, wires=0)
        qml.RY(p.bs_theta, wires=1)
        qml.RZ(p.bs_phi, wires=1)
        # Phase shifters
        for i, phase in enumerate(p.phases):
            qml.RZ(phase, wires=i)
        # Squeezing analogue: rotations
        for i, (r, phi) in enumerate(zip(p.squeeze_r, p.squeeze_phi)):
            qml.RY(r, wires=i)
            qml.RZ(phi, wires=i)
        # Displacement analogue
        for i, (r, phi) in enumerate(zip(p.displacement_r, p.displacement_phi)):
            qml.RY(r, wires=i)
            qml.RZ(phi, wires=i)
        # Kerr analogue
        for i, k in enumerate(p.kerr):
            qml.RZ(k, wires=i)

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Run the QNode for a batch of classical inputs (unused in this simplified demo)."""
        # The circuit is parameter‑independent; we just evaluate the expectation value for each input.
        return np.array([self.qnode() for _ in range(inputs.shape[0])])

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    device=None
) -> qml.QNode:
    """Return a Pennylane QNode that implements the fraud‑detection circuit."""
    detector = FraudDetection(input_params, layers, device=device)
    return detector.qnode

__all__ = ["FraudLayerParameters", "FraudDetection", "build_fraud_detection_program"]
