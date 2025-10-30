"""
Variational quantum circuit for fraud detection using Pennylane.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import pennylane as qml
import pennylane.numpy as np


@dataclass
class FraudLayerParameters:
    """Parameters describing a quantum layer (photonic parameters mapped to qubit gates)."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


class FraudDetectionModel:
    """
    Variational quantum circuit mirroring the photonic fraud detection architecture.
    Supports automatic differentiation and gradient evaluation via Pennylane.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: List[FraudLayerParameters],
        device: str = "default.qubit",
        wires: int = 2,
    ) -> None:
        self.device = qml.device(device, wires=wires)
        self.input_params = input_params
        self.layers = layers
        self.circuit = qml.QNode(self._circuit, self.device, interface="autograd")

    @staticmethod
    def _clip(value: float, bound: float) -> float:
        return max(-bound, min(bound, value))

    def _apply_layer(self, params: FraudLayerParameters, *, clip: bool) -> None:
        # Entangling gate analogous to a beam splitter
        qml.CZ(wires=[0, 1])

        # Phase shifts
        for i, phase in enumerate(params.phases):
            qml.PhaseShift(phase, wires=i)

        # Squeezing → RY rotation (placeholder for continuous‑variable squeezing)
        for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
            qml.RY(self._clip(r if not clip else r, 5.0), wires=i)

        # Displacement → RX rotation
        for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
            qml.RX(self._clip(r if not clip else r, 5.0), wires=i)

        # Kerr nonlinearity → RZ rotation
        for i, k in enumerate(params.kerr):
            qml.RZ(self._clip(k if not clip else k, 1.0), wires=i)

    def _circuit(self, x: np.ndarray) -> np.ndarray:
        # Encode classical data into qubit rotations
        qml.RX(x[0], wires=0)
        qml.RY(x[1], wires=1)

        # Apply layers
        self._apply_layer(self.input_params, clip=False)
        for layer in self.layers:
            self._apply_layer(layer, clip=True)

        # Observable: sum of Pauli‑Z expectation values
        return qml.expval(qml.PauliZ(0)) + qml.expval(qml.PauliZ(1))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the variational circuit."""
        return self.circuit(x)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Return the gradient of the circuit output with respect to parameters."""
        return qml.grad(self.circuit)(x)


__all__ = ["FraudLayerParameters", "FraudDetectionModel"]
