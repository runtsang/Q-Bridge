"""PennyLane implementation of the fraud‑detection circuit.  The same
parameter set is mapped to a variational ansatz that can be trained
jointly with the classical residual network."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pennylane as qml
from pennylane import numpy as np


@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic layer, reused for the quantum
    ansatz."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]
    # New depth hyper‑parameter for the variational ansatz
    depth: int = 1


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def _apply_layer(q: Sequence, params: FraudLayerParameters, *, clip: bool) -> None:
    """Map a photonic layer to a set of PennyLane gates."""
    # Beam‑splitter analogue: two‑qubit rotation
    qml.CRX(params.bs_theta, wires=(0, 1))
    for i, phase in enumerate(params.phases):
        qml.RZ(phase, wires=i)
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        qml.RY(_clip(r, 5.0) if clip else r, wires=i)
        qml.RZ(phi, wires=i)
    # Second beam‑splitter
    qml.CRX(params.bs_theta, wires=(0, 1))
    for i, phase in enumerate(params.phases):
        qml.RZ(phase, wires=i)
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        qml.RY(_clip(r, 5.0) if clip else r, wires=i)
        qml.RZ(phi, wires=i)
    for i, k in enumerate(params.kerr):
        qml.RZ(_clip(k, 1.0) if clip else k, wires=i)


def build_fraud_detection_circuit(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> qml.QNode:
    """Return a PennyLane QNode that evaluates the hybrid fraud‑detection
    circuit.  The circuit is parameterised by the same `FraudLayerParameters`
    objects used in the classical model."""
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs: np.ndarray) -> np.ndarray:
        # Encode classical inputs as rotations
        qml.RY(inputs[0], wires=0)
        qml.RY(inputs[1], wires=1)

        # First layer (unclipped)
        _apply_layer(qml.wires, input_params, clip=False)

        # Subsequent layers (clipped)
        for layer in layers:
            _apply_layer(qml.wires, layer, clip=True)

        # Measurement: expectation of Z on both qubits
        return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

    return circuit


__all__ = ["FraudLayerParameters", "build_fraud_detection_circuit"]
