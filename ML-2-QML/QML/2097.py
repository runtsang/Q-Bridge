"""
Pennylane based variational circuit that mirrors the photonic fraud‑detection layers.
The circuit encodes two classical inputs as rotations, applies a sequence of
parameterised gates derived from FraudLayerParameters, and measures Pauli‑Z
on the first qubit.  The public API mirrors the classical counterpart:
`FraudLayerParameters`, a builder function and a wrapper class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import pennylane as qml
from pennylane import numpy as np


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


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def _apply_layer(
    w0: int,
    w1: int,
    params: FraudLayerParameters,
    *,
    clip: bool,
) -> None:
    """Apply a set of parameterised gates that emulate the photonic layer."""
    # Beam‑splitter equivalent: entangling CZ
    theta = params.bs_theta if not clip else _clip(params.bs_theta, 5)
    qml.CZ(w0, w1)

    # Phase gates
    qml.RZ(params.phases[0], wires=w0)
    qml.RZ(params.phases[1], wires=w1)

    # Squeezing -> Ry rotations (approximated)
    qml.RY(params.squeeze_r[0], wires=w0)
    qml.RY(params.squeeze_r[1], wires=w1)

    # Displacement -> Rx rotations
    qml.RX(params.displacement_r[0], wires=w0)
    qml.RX(params.displacement_r[1], wires=w1)

    # Kerr non‑linearity -> RZ
    qml.RZ(params.kerr[0], wires=w0)
    qml.RZ(params.kerr[1], wires=w1)


def build_fraud_detection_circuit(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> qml.QNode:
    """Return a Pennylane QNode implementing the fraud‑detection circuit."""
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def circuit(inputs: Tuple[float, float]) -> float:
        # Encode inputs as rotations
        qml.RX(inputs[0], wires=0)
        qml.RX(inputs[1], wires=1)

        # Input layer
        _apply_layer(0, 1, input_params, clip=False)

        # Subsequent layers
        for layer_params in layers:
            _apply_layer(0, 1, layer_params, clip=True)

        # Readout
        return qml.expval(qml.PauliZ(0))

    return circuit


class FraudDetectionHybrid:
    """Convenience wrapper around the QNode and prediction routine."""
    def __init__(self, circuit: qml.QNode) -> None:
        self.circuit = circuit

    def predict(self, inputs: Tuple[float, float]) -> float:
        return float(self.circuit(inputs))


__all__ = ["FraudLayerParameters", "build_fraud_detection_circuit", "FraudDetectionHybrid"]
