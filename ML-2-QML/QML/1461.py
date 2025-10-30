"""Quantum fraud detection circuit implemented with Pennylane."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pennylane as qml
import pennylane.numpy as np


@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic layer, reinterpreted for qubits."""

    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


def _clip(value: float, bound: float) -> float:
    """Clamp a scalar to a symmetric interval."""
    return max(-bound, min(bound, value))


class FraudDetector:
    """Quantum fraud detection circuit using Pennylane."""

    def __init__(
        self,
        device: str,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
    ) -> None:
        self.device = qml.device(device, wires=2)
        self.input_params = input_params
        self.layers = layers
        self.qnode = self._build_qnode()

    def _apply_layer(self, params: FraudLayerParameters, clip: bool) -> None:
        """Translate photonic parameters into qubit gates."""
        theta, phi = params.bs_theta, params.bs_phi
        qml.RX(theta, wires=0)
        qml.RX(phi, wires=1)

        for i, phase in enumerate(params.phases):
            qml.RZ(phase, wires=i)

        for i, (r, _) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
            r_clipped = _clip(r, 5.0) if clip else r
            qml.RZZ(r_clipped, wires=i)

        for i, (r, _) in enumerate(zip(params.displacement_r, params.displacement_phi)):
            r_clipped = _clip(r, 5.0) if clip else r
            qml.RX(r_clipped, wires=i)

        for i, k in enumerate(params.kerr):
            k_clipped = _clip(k, 1.0) if clip else k
            qml.RZ(k_clipped, wires=i)

        qml.CZ(wires=[0, 1])

    def _build_qnode(self) -> qml.QNode:
        @qml.qnode(self.device)
        def circuit(x: np.ndarray) -> np.ndarray:
            qml.RY(x[0], wires=0)
            qml.RY(x[1], wires=1)
            self._apply_layer(self.input_params, clip=False)
            for layer in self.layers:
                self._apply_layer(layer, clip=True)
            return [qml.expval(qml.PauliZ(i)) for i in range(2)]

        return circuit

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.qnode(x)


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    device: str = "default.qubit",
) -> qml.QNode:
    """Return a QNode mirroring the classical API."""
    return FraudDetector(device, input_params, layers).qnode


__all__ = ["FraudLayerParameters", "FraudDetector", "build_fraud_detection_program"]
