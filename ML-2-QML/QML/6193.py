"""Quantum photonic fraud detection model using PennyLane."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import pennylane as qml
import numpy as np

@dataclass
class FraudLayerParameters:
    """Parameters for a photonic layer in the quantum model."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]
    variational_depth: int = 1

def _clip(value: float, bound: float) -> float:
    """Clamp a scalar to ``[-bound, bound]``."""
    return max(-bound, min(bound, value))

def _apply_layer(params: FraudLayerParameters, *, clip: bool, variational: np.ndarray | None = None):
    """Apply a single photonic layer to the quantum circuit."""
    qml.BSgate(params.bs_theta, params.bs_phi)
    for phase in params.phases:
        qml.Rgate(phase)
    for r, phi in zip(params.squeeze_r, params.squeeze_phi):
        r = np.clip(r, -5.0, 5.0) if clip else r
        qml.Sgate(r, phi)
    for r, phi in zip(params.displacement_r, params.displacement_phi):
        r = np.clip(r, -5.0, 5.0) if clip else r
        qml.Dgate(r, phi)
    for k in params.kerr:
        k = np.clip(k, -1.0, 1.0) if clip else k
        qml.Kgate(k)
    if variational is not None:
        for v in variational:
            qml.Rgate(v)

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    dev: qml.Device | None = None,
) -> qml.QNode:
    """Create a PennyLane variational circuit for fraud detection."""
    device = dev or qml.device("default.gaussian", wires=2)

    @qml.qnode(device)
    def circuit(*variational_params):
        """Variational photonic circuit."""
        _apply_layer(input_params, clip=False)
        for params, v in zip(layers, variational_params):
            _apply_layer(params, clip=True, variational=v)
        return qml.expval(qml.PauliZ(0))

    return circuit

class FraudDetectionEnhanced:
    """Quantum fraud‑detection model that returns a single qubit expectation value."""
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        dev: qml.Device | None = None,
    ) -> None:
        self.circuit = build_fraud_detection_program(input_params, layers, dev)
        self.variational_params = [
            np.random.uniform(-np.pi, np.pi, size=params.variational_depth)
            for params in layers
        ]

    def predict(self) -> float:
        """Execute the variational circuit and return the expectation value."""
        return float(self.circuit(*self.variational_params))

__all__ = ["FraudLayerParameters", "build_fraud_detection_program", "FraudDetectionEnhanced"]
