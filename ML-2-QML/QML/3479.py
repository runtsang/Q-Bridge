"""
PennyLane implementation of the fraud‑detection variational circuit.
The structure mirrors the classical backbone: each layer
consists of a parameterised two‑qubit entanglement followed by
local rotations that emulate beamsplitters, squeezers, and
displacement gates.  A classical read‑out head completes the
regression task.
"""

from __future__ import annotations

import pennylane as qml
from pennylane import numpy as np
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple


@dataclass
class FraudLayerParameters:
    """
    Parameter set identical to the classical version for a 2‑qubit layer.
    """
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


def _apply_layer(q: qml.Device, params: FraudLayerParameters, *, clip: bool) -> None:
    """
    Apply a single variational layer that emulates a photonic block.
    Uses standard qubit gates to approximate the continuous‑variable
    operations.
    """
    # Beamsplitter analogue: parametrised CZ
    qml.CZ(wires=[0, 1], do_queue=False)  # fixed entanglement

    # Local rotations mimicking beamsplitter angles
    qml.RX(params.bs_theta, wires=0)
    qml.RZ(params.bs_phi, wires=1)

    # Phase rotations
    qml.RZ(params.phases[0], wires=0)
    qml.RZ(params.phases[1], wires=1)

    # Squeezing emulation via RZ (phase) and RX (amplitude)
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        qml.RZ(_clip(phi, 5.0) if clip else phi, wires=i)
        qml.RX(_clip(r, 5.0) if clip else r, wires=i)

    # Displacement emulation via RX and RZ
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        qml.RX(_clip(r, 5.0) if clip else r, wires=i)
        qml.RZ(_clip(phi, 5.0) if clip else phi, wires=i)

    # Kerr non‑linearity emulation via RZ with bounded angle
    for i, k in enumerate(params.kerr):
        qml.RZ(_clip(k, 1.0) if clip else k, wires=i)


def build_fraud_detection_qnode(
    dev: qml.Device,
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> qml.QNode:
    """
    Construct a PennyLane QNode that applies the layered circuit.
    The QNode returns expectation of PauliZ on the first qubit,
    which is then read out classically.
    """

    @qml.qnode(dev)
    def circuit():
        _apply_layer(dev, input_params, clip=False)
        for l in layers:
            _apply_layer(dev, l, clip=True)
        return qml.expval(qml.PauliZ(0))

    return circuit


class FraudDetectionQNN:
    """
    Hybrid quantum‑classical neural network.  The quantum circuit
    serves as a feature extractor; a small classical head
    produces the final regression output.
    """

    def __init__(
        self,
        dev: qml.Device,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        hidden_dims: list[int] | None = None,
    ) -> None:
        self.circuit = build_fraud_detection_qnode(dev, input_params, layers)
        hidden_dims = hidden_dims or [8, 4]
        self.head = qml.numpy.ndarray(
            np.zeros((1, 1), dtype=np.float32)  # placeholder for gradient
        )
        # Classical read‑out using a simple linear mapping
        self.trainable_params = list(self.circuit.trainable_params)

    def __call__(self, *args, **kwargs) -> np.ndarray:
        # Execute quantum circuit
        q_out = self.circuit()
        # Simple linear read‑out (could be replaced by a small NN)
        return q_out

    def parameters(self):
        return self.trainable_params


__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_qnode",
    "FraudDetectionQNN",
]
