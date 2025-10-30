"""Quantum fraud detection model using PennyLane and StrawberryFields backend.

The model implements a variational photonic circuit whose parameters are
wrapped in `FraudLayerParameters`.  Each layer consists of a beamsplitter,
phase shifters, squeezers, displacements and Kerr gates.  The input
vector is encoded into the initial displacement of the two modes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pennylane as qml
import pennylane.numpy as pnp
from pennylane import qnode


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


class FraudDetectionModel:
    """Variational photonic fraud detection model."""

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        device: qml.Device | None = None,
        cutoff_dim: int = 10,
    ) -> None:
        self.input_params = input_params
        self.layers = list(layers)
        self.device = device or qml.device("strawberryfields.fock", wires=2, cutoff_dim=cutoff_dim)
        self._build_qnode()

    def _apply_layer(self, params: FraudLayerParameters, clip: bool) -> None:
        qml.BSgate(params.bs_theta, params.bs_phi, wires=[0, 1])
        for i, phase in enumerate(params.phases):
            qml.Rgate(phase, wires=i)
        for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
            r_clip = r if not clip else _clip(r, 5)
            qml.Sgate(r_clip, phi, wires=i)
        qml.BSgate(params.bs_theta, params.bs_phi, wires=[0, 1])
        for i, phase in enumerate(params.phases):
            qml.Rgate(phase, wires=i)
        for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
            r_clip = r if not clip else _clip(r, 5)
            qml.Dgate(r_clip, phi, wires=i)
        for i, k in enumerate(params.kerr):
            k_clip = k if not clip else _clip(k, 1)
            qml.Kgate(k_clip, wires=i)

    def _build_qnode(self) -> None:
        @qml.qnode(self.device, interface="autograd")
        def circuit(x: Sequence[float]) -> pnp.ndarray:
            # Encode input data as displacements of the two modes
            for i, val in enumerate(x):
                qml.Dgate(val, 0.0, wires=i)
            self._apply_layer(self.input_params, clip=False)
            for layer in self.layers:
                self._apply_layer(layer, clip=True)
            # Measurement: expectation value of Z on mode 0
            return qml.expval(qml.PauliZ(0))
        self.circuit = circuit

    def forward(self, x: Sequence[float]) -> pnp.ndarray:
        """Return the raw measurement expectation value."""
        return self.circuit(x)

    def predict(self, x: Sequence[float]) -> pnp.ndarray:
        """Map the expectation value to a probability in [0, 1]."""
        return 0.5 * (self.forward(x) + 1.0)

__all__ = ["FraudLayerParameters", "FraudDetectionModel"]
