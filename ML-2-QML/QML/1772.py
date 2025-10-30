"""Quantum fraud detection model using PennyLane."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List
import pennylane as qml
import pennylane.numpy as np


@dataclass
class FraudLayerParameters:
    """Parameter set for a single variational layer."""
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


def _params_to_array(params: FraudLayerParameters) -> np.ndarray:
    return np.array(
        [
            params.bs_theta,
            params.bs_phi,
            *params.phases,
            *params.squeeze_r,
            *params.squeeze_phi,
            *params.displacement_r,
            *params.displacement_phi,
            *params.kerr,
        ]
    )


def _apply_layer_qml(params: np.ndarray, clip: bool) -> None:
    """Apply a single variational layer to the quantum circuit."""
    bs_theta, bs_phi, p0, p1, s0r, s0phi, s1r, s1phi, d0r, d0phi, d1r, d1phi, k0, k1 = params

    # Beam splitter equivalent: rotation + entanglement
    qml.RX(bs_theta, wires=0)
    qml.RZ(bs_phi, wires=1)
    qml.CZ(wires=[0, 1])

    # Phase shifts
    qml.RZ(p0, wires=0)
    qml.RZ(p1, wires=1)

    # Squeezing (mapped to rotations)
    qml.RX(_clip(s0r, 5) if clip else s0r, wires=0)
    qml.RX(_clip(s1r, 5) if clip else s1r, wires=1)

    # Displacement (mapped to rotations)
    qml.RZ(_clip(d0r, 5) if clip else d0r, wires=0)
    qml.RZ(_clip(d1r, 5) if clip else d1r, wires=1)

    # Kerr non‑linearity is omitted in the qubit model but kept for parameter consistency


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> qml.QNode:
    """Create a PennyLane QNode representing the fraud detection circuit."""
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def circuit(*param_arrays: np.ndarray) -> np.ndarray:
        # Flatten layers into a list of parameter arrays
        all_params = [np.array(_params_to_array(input_params))] + list(param_arrays)
        for idx, params in enumerate(all_params):
            _apply_layer_qml(params, clip=(idx > 0))
        # Two‑output regression: expectation values of Z on each qubit
        return np.array([qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))])

    return circuit


__all__ = ["FraudLayerParameters", "build_fraud_detection_program"]
