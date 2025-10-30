"""Quantum photonic fraud detection circuit implemented with PennyLane.

The class `FraudDetectionModel` builds a variational circuit that mirrors
the classical architecture.  It uses a photonic device (Strawberry
Fields) and measures the total photon number as a proxy for the
output.  The interface is identical to the classical model, enabling
side‑by‑side experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import pennylane as qml
import numpy as np


@dataclass
class FraudLayerParameters:
    """Parameters for a single photonic layer."""
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


class FraudDetectionModel:
    """Variational photonic fraud‑detection circuit."""

    def __init__(self, dev: qml.Device | None = None, wires: int = 2) -> None:
        self.dev = dev or qml.device("pennylane.plugins.SF", wires=wires)
        self.qnode = qml.QNode(self._circuit, self.dev, interface="autograd")

    def _circuit(
        self,
        params: List[FraudLayerParameters],
        x: np.ndarray,
    ) -> float:
        # encode the classical input into coherent states
        for i, val in enumerate(x):
            qml.Displacement(val, 0.0) | i

        # apply the layered photonic circuit
        for layer in params:
            # beam splitter
            qml.BSgate(layer.bs_theta, layer.bs_phi) | (0, 1)
            # phase shifts
            for j, phase in enumerate(layer.phases):
                qml.Rgate(phase) | j
            # squeezing
            for j, (r, phi) in enumerate(zip(layer.squeeze_r, layer.squeeze_phi)):
                qml.Sgate(_clip(r, 5.0), phi) | j
            # second beam splitter (symmetry)
            qml.BSgate(layer.bs_theta, layer.bs_phi) | (0, 1)
            # displacement
            for j, (r, phi) in enumerate(zip(layer.displacement_r, layer.displacement_phi)):
                qml.Dgate(_clip(r, 5.0), phi) | j
            # Kerr
            for j, k in enumerate(layer.kerr):
                qml.Kgate(_clip(k, 1.0)) | j

        # measurement: total photon number
        return qml.expval(qml.NumberOperator(0)) + qml.expval(qml.NumberOperator(1))

    def forward(
        self,
        params: List[FraudLayerParameters],
        x: np.ndarray,
    ) -> np.ndarray:
        """Run the variational circuit and return the expectation value."""
        return np.array([self.qnode(params, x)])

    def to_classical_params(self, params: List[FraudLayerParameters]) -> List[FraudLayerParameters]:
        """
        Identity mapping: the quantum parameters are already in the
        format expected by the classical model.  This method is kept
        for API symmetry with the classical counterpart.
        """
        return params
