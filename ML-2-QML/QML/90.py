"""Quantum-enhanced fraud detection model using PennyLane."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pennylane as qml
import numpy as np


@dataclass
class FraudLayerParameters:
    """Parameters describing a photonic-like layer, extended with qubit depth."""

    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]
    q_depth: int = 1  # number of variational layers per photonic layer


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def _apply_layer(wires: Sequence[int], params: FraudLayerParameters, *, clip: bool) -> None:
    # Entangling layer mimicking a balanced beamsplitter
    for i in range(len(wires) - 1):
        qml.CNOT(wires=[wires[i], wires[i + 1]])
    # Single-qubit rotations
    for i, phase in enumerate(params.phases):
        qml.PhaseShift(phase, wires=wires[i])
    # Squeezing via Z rotations (approximation)
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        angle = _clip(r, 5.0) if clip else r
        qml.RZ(angle, wires=wires[i])
    # Displacement via X rotations (approximation)
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        angle = _clip(r, 5.0) if clip else r
        qml.RX(angle, wires=wires[i])
    # Kerr via Z rotations
    for i, k in enumerate(params.kerr):
        angle = _clip(k, 1.0) if clip else k
        qml.RZ(angle, wires=wires[i])


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> qml.QNode:
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev, interface="auto")
    def circuit(inputs: np.ndarray) -> np.ndarray:
        # Encode classical inputs into qubit states via RX rotations
        qml.RX(inputs[0], wires=0)
        qml.RX(inputs[1], wires=1)

        _apply_layer([0, 1], input_params, clip=False)
        for layer in layers:
            _apply_layer([0, 1], layer, clip=True)

        # Measure expectation values of Pauli Z on each qubit
        return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

    return circuit


class FraudDetectionHybrid:
    """Quantum circuit wrapper returning a single scalar output."""

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
    ) -> None:
        self.circuit = build_fraud_detection_program(input_params, layers)

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        z_vals = self.circuit(inputs)
        # Simple linear readout followed by tanh activation
        return np.tanh(z_vals[0] + z_vals[1])


__all__ = ["FraudLayerParameters", "build_fraud_detection_program", "FraudDetectionHybrid"]
