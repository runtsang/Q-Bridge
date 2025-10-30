"""Quantum fraud detection circuit implemented with Pennylane."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pennylane as qml
import numpy as np
import torch


@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic layer used in the quantum circuit."""

    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


def _clip(value: float, bound: float) -> float:
    """Clamp values to a safe range before they are used in the circuit."""
    return max(-bound, min(bound, value))


def _apply_layer(
    wires: Sequence[int], params: FraudLayerParameters, *, clip: bool
) -> None:
    """Apply the photonic layer gates to the specified wires."""
    # Beam‑splitter
    qml.BSgate(params.bs_theta, params.bs_phi, wires=[wires[0], wires[1]])
    # Phase shifters
    for i, phase in enumerate(params.phases):
        qml.Rgate(phase, wires=wires[i])
    # Squeezing
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        qml.Sgate(r if not clip else _clip(r, 5.0), phi, wires=wires[i])
    # Second beam‑splitter
    qml.BSgate(params.bs_theta, params.bs_phi, wires=[wires[0], wires[1]])
    # Repeat phases
    for i, phase in enumerate(params.phases):
        qml.Rgate(phase, wires=wires[i])
    # Displacement
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        qml.Dgate(r if not clip else _clip(r, 5.0), phi, wires=wires[i])
    # Kerr
    for i, k in enumerate(params.kerr):
        qml.Kgate(k if not clip else _clip(k, 1.0), wires=wires[i])


def build_fraud_detection_circuit(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    *,
    shots: int | None = None,
) -> qml.QNode:
    """Create a Pennylane QNode that realizes the fraud‑detection photonic
    circuit and returns the expectation of a Pauli‑Z observable."""
    dev = qml.device("default.qubit", wires=2, shots=shots)

    @qml.qnode(dev, interface="torch")
    def circuit(x: torch.Tensor) -> torch.Tensor:
        # The input vector x is not used directly; it is kept to preserve
        # a consistent interface with the classical model.
        _apply_layer([0, 1], input_params, clip=False)
        for layer in layers:
            _apply_layer([0, 1], layer, clip=True)
        return qml.expval(qml.PauliZ(0))

    return circuit


class FraudDetectionHybrid:
    """Quantum fraud detection model that outputs a probability via a variational circuit."""

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        *,
        shots: int | None = None,
    ) -> None:
        self.circuit = build_fraud_detection_circuit(input_params, layers, shots=shots)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.circuit(x)


__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_circuit",
    "FraudDetectionHybrid",
]
