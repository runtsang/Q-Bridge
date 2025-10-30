"""Quantum analogue of the fraud‑detection circuit using Pennylane."""

from __future__ import annotations

import pennylane as qml
from pennylane import numpy as np
from dataclasses import dataclass
from typing import Iterable, Sequence

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

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _apply_layer(wires: Sequence[int], params: FraudLayerParameters, *, clip: bool) -> None:
    # Beam‑splitter analogue via single‑qubit rotations (placeholder for a real BS gate)
    qml.Rot(params.bs_theta, params.bs_phi, 0, wires=wires[0])
    qml.Rot(params.bs_theta, params.bs_phi, 0, wires=wires[1])

    # Phase shifts
    for i, phase in enumerate(params.phases):
        qml.PhaseShift(phase, wires=wires[i])

    # Squeezing → RY rotations
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        r_clipped = _clip(r, 5) if clip else r
        qml.RY(r_clipped, wires=wires[i])
        qml.PhaseShift(phi, wires=wires[i])

    # Displacement → RX rotations
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        r_clipped = _clip(r, 5) if clip else r
        qml.RX(r_clipped, wires=wires[i])
        qml.PhaseShift(phi, wires=wires[i])

    # Kerr → RZ rotations
    for i, k in enumerate(params.kerr):
        k_clipped = _clip(k, 1) if clip else k
        qml.RZ(k_clipped, wires=wires[i])

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> qml.QNode:
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev, interface="torch")
    def circuit(x: np.ndarray):
        # Encode classical features via rotations
        qml.RY(x[0], wires=0)
        qml.RY(x[1], wires=1)

        _apply_layer([0, 1], input_params, clip=False)
        for layer in layers:
            _apply_layer([0, 1], layer, clip=True)

        # Measurement of Pauli‑Z on each qubit
        return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

    return circuit

class FraudDetectionHybrid:
    """Quantum‑classical hybrid fraud‑detection model built with Pennylane."""

    def __init__(self, input_params: FraudLayerParameters, layers: Iterable[FraudLayerParameters]) -> None:
        self.circuit = build_fraud_detection_program(input_params, layers)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.circuit(x)

__all__ = ["FraudLayerParameters", "FraudDetectionHybrid"]
