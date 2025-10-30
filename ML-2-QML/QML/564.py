"""Variational fraud detection circuit built with Pennylane.

The circuit mirrors the photonic structure by embedding the parameters of each
layer into a series of RX, RZ, and CNOT gates.  A residual‑style shortcut is
implemented by concatenating the expectation values of the two qubits.  The
`FraudDetectionHybrid` class exposes a `qnode` that can be differentiated
with respect to the layer parameters, enabling hybrid training with any
autograd‑compatible loss function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pennylane as qml
import pennylane.numpy as np

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

class FraudDetectionHybrid:
    """Variational hybrid fraud detection circuit using Pennylane."""

    def __init__(self, dev: qml.Device, input_params: FraudLayerParameters, layers: Iterable[FraudLayerParameters]):
        self.dev = dev
        self.input_params = input_params
        self.layers = list(layers)

        # Build the QNode with autograd interface
        self.qnode = qml.QNode(self._circuit, dev, interface="autograd")

    def _circuit(self, *params):
        # params contains flattened parameter list for each layer
        param_iter = iter(params)

        # Input embedding – use displacement_r as rotation angles
        for i, r in enumerate(self.input_params.displacement_r):
            qml.RX(r, wires=i)

        # Loop over layers
        for layer in self.layers:
            # Encode layer parameters into rotations
            for i, r in enumerate(layer.displacement_r):
                qml.RX(r, wires=i)
            qml.CNOT(wires=[0, 1])
            for i, theta in enumerate([layer.bs_theta, layer.bs_phi]):
                qml.RZ(theta, wires=i)
            for i, k in enumerate(layer.kerr):
                qml.RX(k, wires=i)

        # Residual‑style measurement: sum of Pauli‑Z expectations
        return qml.expval(qml.PauliZ(0)) + qml.expval(qml.PauliZ(1))

    def __call__(self, *args):
        return self.qnode(*args)

__all__ = ["FraudLayerParameters", "FraudDetectionHybrid"]
