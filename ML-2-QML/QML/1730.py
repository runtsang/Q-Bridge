"""Pennylane based variational fraud‑detection circuit.

The original seed used Strawberry Fields to construct a photonic
ansatz.  Here we replace the static circuit with a fully
parameterised variational QNode that can be trained with
automatic differentiation.  The same `FraudLayerParameters`
dataclass is reused so that the classical and quantum
implementations share a common hyper‑parameter schema.

Key extensions:
* A two‑qubit ansatz with RY, RZ, and CZ entangling gates.
* All photonic parameters are mapped to rotation angles or
  displacement‑like gates in the qubit picture.
* The QNode exposes an expectation value of `PauliZ(0)` as the
  network output, mimicking the scalar output of the classical
  model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pennylane as qml
import numpy as np


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


class FraudDetectionModel:
    """Variational fraud‑detection circuit."""

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        device: str = "default.qubit",
        shots: int = 1000,
    ) -> None:
        self.device = qml.device(device, wires=2, shots=shots)
        self.input_params = input_params
        self.layers = list(layers)

        # Build a QNode that captures all static parameters
        self.qnode = self._build_qnode()

    def _build_qnode(self) -> qml.QNode:
        @qml.qnode(self.device, interface="autograd")
        def circuit(inputs: np.ndarray) -> float:
            # Encode inputs as rotation angles
            qml.RY(inputs[0], wires=0)
            qml.RZ(inputs[1], wires=1)

            # Apply input layer
            self._apply_photonic_layer(
                self.input_params, clip=False, wires=range(2)
            )

            # Apply subsequent layers
            for layer in self.layers:
                self._apply_photonic_layer(layer, clip=True, wires=range(2))

            # Return scalar output
            return qml.expval(qml.PauliZ(0))

        return circuit

    def _apply_photonic_layer(
        self,
        params: FraudLayerParameters,
        clip: bool,
        wires: Sequence[int],
    ) -> None:
        """Map photonic parameters to qubit rotations."""
        # Beam‑splitter angles -> RY rotations
        for w, angle in zip(wires, (params.bs_theta, params.bs_phi)):
            qml.RY(angle, wires=w)

        # Phases -> RZ rotations
        for w, phase in zip(wires, params.phases):
            qml.RZ(phase, wires=w)

        # Squeezing -> RZ with modified angle
        for w, (r, phi) in zip(wires, zip(params.squeeze_r, params.squeeze_phi)):
            r_adj = r if not clip else np.clip(r, -5.0, 5.0)
            qml.RZ(phi + r_adj, wires=w)

        # Displacement -> RY
        for w, (r, phi) in zip(wires, zip(params.displacement_r, params.displacement_phi)):
            r_adj = r if not clip else np.clip(r, -5.0, 5.0)
            qml.RY(r_adj, wires=w)

        # Kerr non‑linearity -> RZ
        for w, k in zip(wires, params.kerr):
            k_adj = k if not clip else np.clip(k, -1.0, 1.0)
            qml.RZ(k_adj, wires=w)

        # Entangling layer to mimic photonic entanglement
        qml.CZ(wires[0], wires[1])

    def forward(self, inputs: np.ndarray) -> float:
        """Evaluate the circuit for a given 2‑dimensional input vector."""
        return float(self.qnode(inputs))

    def parameters(self) -> Sequence[tuple]:
        """Return all trainable parameters of the QNode."""
        return self.qnode.parameters


__all__ = ["FraudLayerParameters", "FraudDetectionModel"]
