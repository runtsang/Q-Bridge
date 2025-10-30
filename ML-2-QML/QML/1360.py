"""PennyLane CV variational circuit for fraud detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pennylane as qml


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
    return max(-bound, min(bound, value))


def _apply_layer(wires: Sequence[int], params: FraudLayerParameters, clip: bool) -> None:
    """Apply a photonic layer to the given wires."""
    qml.BSgate(params.bs_theta, params.bs_phi, wires=wires)
    for i, phase in enumerate(params.phases):
        qml.Rgate(phase, wires=wires[i])
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        qml.Sgate(r if not clip else _clip(r, 5), phi, wires=wires[i])
    qml.BSgate(params.bs_theta, params.bs_phi, wires=wires)
    for i, phase in enumerate(params.phases):
        qml.Rgate(phase, wires=wires[i])
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        qml.Dgate(r if not clip else _clip(r, 5), phi, wires=wires[i])
    for i, k in enumerate(params.kerr):
        qml.Kgate(k if not clip else _clip(k, 1), wires=wires[i])


def build_fraud_detection_circuit(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    device: qml.Device | None = None,
) -> qml.QNode:
    """
    Build a PennyLane QNode that implements the hybrid fraud detection
    photonic circuit.  The circuit prepares an arbitrary two‑mode state
    (passed as the first argument to the QNode), applies the input layer
    without clipping, then each subsequent layer with clipping, and
    finally measures the Pauli‑Z expectation value of the first mode.
    """
    if device is None:
        device = qml.device("default.qubit", wires=2)

    @qml.qnode(device)
    def circuit(inputs: Sequence[float]) -> float:
        qml.StatePreparation(inputs, wires=range(2))
        _apply_layer(range(2), input_params, clip=False)
        for layer in layers:
            _apply_layer(range(2), layer, clip=True)
        return qml.expval(qml.PauliZ(0))

    return circuit


__all__ = ["FraudLayerParameters", "build_fraud_detection_circuit"]
