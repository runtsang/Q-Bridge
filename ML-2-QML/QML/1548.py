"""Quantum photonic fraud‑detection circuit using Pennylane’s Strawberry‑Fields plugin.

The hybrid circuit mirrors the classical network and returns a Pauli‑Z
expectation value that can be interpreted as a fraud probability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pennylane as qml
import pennylane.strawberryfields as sf
from pennylane import QNode, Device


@dataclass
class FraudLayerParameters:
    """Parameters describing a photonic layer."""

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


def _apply_layer(
    wires: qml.Wires,
    params: FraudLayerParameters,
    *,
    clip: bool,
) -> None:
    """Apply a photonic layer to the given wires."""
    # Beam‑splitter
    qml.BSgate(params.bs_theta, params.bs_phi, wires=wires)
    # Rotations
    for i, phase in enumerate(params.phases):
        qml.Rgate(phase, wires=wires[i])
    # Squeezing
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        qml.Sgate(r if not clip else _clip(r, 5), phi, wires=wires[i])
    # Second beam‑splitter
    qml.BSgate(params.bs_theta, params.bs_phi, wires=wires)
    # Rotations again
    for i, phase in enumerate(params.phases):
        qml.Rgate(phase, wires=wires[i])
    # Displacement
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        qml.Dgate(r if not clip else _clip(r, 5), phi, wires=wires[i])
    # Kerr
    for i, k in enumerate(params.kerr):
        qml.Kgate(k if not clip else _clip(k, 1), wires=wires[i])


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    *,
    device: Device | None = None,
) -> QNode:
    """Create a Pennylane QNode that implements the hybrid fraud‑detection circuit."""
    if device is None:
        device = qml.Device("strawberryfields", wires=2)

    @qml.qnode(device)
    def circuit():
        _apply_layer(qml.Wires([0, 1]), input_params, clip=False)
        for layer in layers:
            _apply_layer(qml.Wires([0, 1]), layer, clip=True)
        # Return a simple observable; more complex post‑processing
        # can be added by the user.
        return qml.expval(qml.PauliZ(0))

    return circuit


class FraudDetectionModel:
    """Wrapper around the Pennylane photonic circuit."""

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        *,
        device: Device | None = None,
    ) -> None:
        self.input_params = input_params
        self.layers = list(layers)
        self.qnode = build_fraud_detection_program(
            input_params, self.layers, device=device
        )

    def __call__(self) -> float:
        """Run the circuit and return the measured expectation value."""
        return self.qnode()


__all__ = [
    "FraudLayerParameters",
    "FraudDetectionModel",
    "build_fraud_detection_program",
]
