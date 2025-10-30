"""Quantum fraud detection model using Pennylane.

The circuit encodes the classical input into rotations and applies a
trainable variational layer inspired by the photonic operations.  The
output is the expectation value of Pauli‑Z on a single qubit, which is
passed through a sigmoid to produce a fraud probability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import pennylane as qml
import pennylane.numpy as np


@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]
    dropout: float = 0.0  # placeholder for compatibility


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def _apply_layer(dev: qml.Device, params: FraudLayerParameters, *, clip: bool) -> None:
    # Encode input using rotations
    qml.RY(params.bs_theta, wires=0)
    qml.RY(params.bs_phi, wires=1)
    for i, phase in enumerate(params.phases):
        qml.RZ(phase, wires=i)
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        qml.RZ(_clip(r, 5) if clip else r, wires=i)
    qml.CNOT(wires=[0, 1])
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        qml.RZ(_clip(r, 5) if clip else r, wires=i)
    for i, k in enumerate(params.kerr):
        qml.RZ(_clip(k, 1) if clip else k, wires=i)


def build_fraud_detection_circuit(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    dev: qml.Device,
) -> qml.QNode:
    """Return a QNode that implements the fraud‑detection circuit."""
    @qml.qnode(dev)
    def circuit(x: np.ndarray) -> np.ndarray:
        qml.BasisState(x, wires=[0, 1])
        _apply_layer(dev, input_params, clip=False)
        for layer in layers:
            _apply_layer(dev, layer, clip=True)
        return qml.expval(qml.PauliZ(0))
    return circuit


class FraudDetectionHybrid:
    """Quantum fraud detection model with a Pennylane QNode."""
    def __init__(self, dev: qml.Device, input_params: FraudLayerParameters, layers: Iterable[FraudLayerParameters]):
        self.circuit = build_fraud_detection_circuit(input_params, layers, dev)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the model on a 2‑element input."""
        z = self.circuit(x)
        return 1 / (1 + np.exp(-z))  # sigmoid


__all__ = ["FraudLayerParameters", "build_fraud_detection_circuit", "FraudDetectionHybrid"]
