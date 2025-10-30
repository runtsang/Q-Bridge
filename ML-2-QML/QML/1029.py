"""PennyLane variational circuit for fraud detection, mirroring the photonic layer structure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pennylane as qml
from pennylane import numpy as np


@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic layer."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


def _clip(value: float, bound: float) -> float:
    """Clamp a scalar to the range [-bound, bound]."""
    return max(-bound, min(bound, value))


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> qml.QNode:
    """Return a PennyLane QNode that implements the fraud‑detection circuit."""
    dev = qml.device("default.qubit", wires=2)

    # Convert the iterable to a list for repeated indexing
    layer_list = list(layers)

    @qml.qnode(dev, interface="torch")
    def circuit():
        # Apply the first (unclipped) layer
        _apply_layer(0, input_params, clip=False)

        # Apply subsequent clipped layers
        for idx, layer in enumerate(layer_list, start=1):
            _apply_layer(idx, layer, clip=True)

        # Expectation values of PauliZ on each qubit
        return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]

    def _apply_layer(idx: int, params: FraudLayerParameters, *, clip: bool) -> None:
        """Map photonic operations to qubit gates."""
        # Beamsplitter → parameterised RY followed by CNOT for entanglement
        theta = params.bs_theta if not clip else _clip(params.bs_theta, 5.0)
        phi = params.bs_phi if not clip else _clip(params.bs_phi, 5.0)
        qml.RY(theta, wires=0)
        qml.RY(phi, wires=1)
        qml.CNOT(wires=[0, 1])

        # Phase rotations
        for i, phase in enumerate(params.phases):
            qml.RZ(phase, wires=i)

        # Squeezing → RZ with magnitude r (clipped)
        for i, (r, phi_s) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
            r_clipped = r if not clip else _clip(r, 5.0)
            qml.RZ(r_clipped, wires=i)

        # Displacement → RZ with magnitude r (clipped)
        for i, (r, phi_d) in enumerate(zip(params.displacement_r, params.displacement_phi)):
            r_clipped = r if not clip else _clip(r, 5.0)
            qml.RZ(r_clipped, wires=i)

        # Kerr non‑linearity → phase shift (RZ)
        for i, k in enumerate(params.kerr):
            k_clipped = k if not clip else _clip(k, 1.0)
            qml.RZ(k_clipped, wires=i)

    return circuit


__all__ = ["build_fraud_detection_program"]
