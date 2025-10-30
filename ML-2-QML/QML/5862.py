"""FraudDetectionAdvanced: Quantum version using Pennylane VQE ansatz with parameter sharing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pennylane as qml
import pennylane.numpy as np


@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic layer, mapped to a variational circuit."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]
    num_layers: int = 1          # depth of the ansatz
    use_ancilla: bool = False    # flag for future ancilla usage


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def _apply_layer(wires: Sequence[int], params: FraudLayerParameters, *, clip: bool) -> None:
    # Beam‑splitter equivalent with two rotations
    qml.Rot(params.bs_theta, params.bs_phi, 0.0) | wires[0]
    qml.Rot(0.0, params.bs_phi, params.bs_theta) | wires[1]
    for i, phase in enumerate(params.phases):
        qml.RZ(phase) | wires[i]
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        qml.RZ(_clip(r, 5.0) if clip else r) | wires[i]
    # Entangling block
    qml.CNOT(wires[0], wires[1])
    for i, phase in enumerate(params.phases):
        qml.RZ(phase) | wires[i]
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        qml.RZ(_clip(r, 5.0) if clip else r) | wires[i]
    for i, k in enumerate(params.kerr):
        qml.RZ(_clip(k, 1.0) if clip else k) | wires[i]


def build_fraud_detection_circuit(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> qml.QNode:
    """Return a Pennylane QNode implementing the fraud detection circuit."""
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def circuit(inputs: Sequence[float]) -> float:
        # Encode classical inputs as rotations
        qml.RZ(inputs[0]) | 0
        qml.RZ(inputs[1]) | 1
        _apply_layer([0, 1], input_params, clip=False)
        for layer in layers:
            _apply_layer([0, 1], layer, clip=True)
        # Measurement expectation value of PauliZ on the first qubit
        return qml.expval(qml.PauliZ(0))

    return circuit


__all__ = ["FraudLayerParameters", "build_fraud_detection_circuit"]
