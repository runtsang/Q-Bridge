"""Quantum variational fraud‑detection circuit implemented with PennyLane."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pennylane as qml
import numpy as np


@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic layer, reused for the QML model."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


def _clip(value: float, bound: float) -> float:
    """Clamp a scalar to the interval [−bound, bound]."""
    return max(-bound, min(bound, value))


def _apply_layer(q: qml.Device, params: FraudLayerParameters, *, clip: bool) -> None:
    """Add a variational layer that mirrors the photonic operations."""
    # Beam‑splitter analogue: entangle the two qubits
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 0])

    # Phase rotations
    for i, phase in enumerate(params.phases):
        qml.RZ(phase, wires=i)

    # Squeezing analogue: parameterised RY rotations
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        angle = r if not clip else _clip(r, 5.0)
        qml.RY(angle, wires=i)
        qml.RZ(phi, wires=i)

    # Displacement analogue: further rotations
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        angle = r if not clip else _clip(r, 5.0)
        qml.RX(angle, wires=i)
        qml.RZ(phi, wires=i)

    # Kerr non‑linearity analogue: parameterised phase shift
    for i, k in enumerate(params.kerr):
        angle = k if not clip else _clip(k, 1.0)
        qml.RZ(angle, wires=i)


class FraudDetector:
    """
    PennyLane variational circuit that emulates the photonic fraud‑detection model.
    The circuit returns the expectation value of PauliZ on wire 0.
    """
    def __init__(self,
                 input_params: FraudLayerParameters,
                 layers: Iterable[FraudLayerParameters],
                 device_name: str = "default.qubit",
                 wires: int = 2) -> None:
        self.device = qml.device(device_name, wires=wires)
        self.input_params = input_params
        self.layers = list(layers)

        @qml.qnode(self.device)
        def circuit() -> float:
            _apply_layer(self.device, self.input_params, clip=False)
            for layer in self.layers:
                _apply_layer(self.device, layer, clip=True)
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def __call__(self) -> float:
        """Evaluate the circuit and return the expectation value."""
        return self.circuit()


def build_fraud_detection_qprogram(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> FraudDetector:
    """Convenience wrapper to instantiate the quantum FraudDetector."""
    return FraudDetector(input_params, layers)


__all__ = ["FraudLayerParameters", "FraudDetector"]
