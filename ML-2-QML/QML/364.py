"""Hybrid fraud detection model implemented with PennyLane variational circuit."""

import pennylane as qml
from pennylane import numpy as np
from dataclasses import dataclass
from typing import Iterable, Sequence, Callable


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
    # Entangling gate (analogous to BSgate)
    qml.CNOT(wires=[wires[0], wires[1]])  # simple entangler
    # Single‑mode rotations
    for i, phase in enumerate(params.phases):
        qml.RZ(phase, wires=wires[i])
    # Squeezing and displacement approximated with rotations
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        r_val = r if not clip else _clip(r, 5)
        phi_val = phi
        qml.RX(r_val, wires=wires[i])  # map squeeze to RX
        qml.RZ(phi_val, wires=wires[i])  # map phi to RZ
    # Additional entanglement
    qml.CNOT(wires=[wires[0], wires[1]])
    # Displacement approximated with rotations
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        r_val = r if not clip else _clip(r, 5)
        phi_val = phi
        qml.RX(r_val, wires=wires[i])
        qml.RZ(phi_val, wires=wires[i])
    # Kerr nonlinearity approximated with RZZ
    for i, k in enumerate(params.kerr):
        k_val = k if not clip else _clip(k, 1)
        qml.RZ(k_val, wires=wires[i])


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a PennyLane QNode that implements the fraud detection circuit."""
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def circuit(x: np.ndarray) -> np.ndarray:
        # Encode input as rotation angles
        qml.RX(x[0], wires=0)
        qml.RX(x[1], wires=1)

        # Apply input layer
        _apply_layer([0, 1], input_params, clip=False)

        # Apply subsequent layers
        for layer in layers:
            _apply_layer([0, 1], layer, clip=True)

        # Measure expectation of PauliZ on both wires and concatenate
        return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

    return circuit


class FraudDetectionHybrid:
    """Wrapper that exposes the PennyLane circuit with a callable interface."""

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
    ) -> None:
        self.circuit = build_fraud_detection_program(input_params, layers)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Execute the variational circuit on input data ``x``."""
        return self.circuit(x)


__all__ = ["FraudLayerParameters", "build_fraud_detection_program", "FraudDetectionHybrid"]
