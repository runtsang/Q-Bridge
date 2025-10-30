"""Quantum photonic fraud detection circuit using PennyLane's Gaussian device."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import pennylane as qml
import numpy as np
from pennylane import numpy as pnp


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


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def _apply_layer(
    params: FraudLayerParameters, clip: bool = False
) -> List[qml.operation.Operation]:
    ops: List[qml.operation.Operation] = []

    # Beam splitter
    ops.append(qml.BSgate(params.bs_theta, params.bs_phi, wires=[0, 1]))

    # Phase shifters
    for i, phase in enumerate(params.phases):
        ops.append(qml.Rgate(phase, wires=i))

    # Squeezing (clipped if requested)
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        ops.append(
            qml.Sgate(_clip(r, 5) if clip else r, phi, wires=i)
        )

    # Second beam splitter
    ops.append(qml.BSgate(params.bs_theta, params.bs_phi, wires=[0, 1]))

    # Phase shifters again
    for i, phase in enumerate(params.phases):
        ops.append(qml.Rgate(phase, wires=i))

    # Displacement (clipped if requested)
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        ops.append(
            qml.Dgate(_clip(r, 5) if clip else r, phi, wires=i)
        )

    # Kerr nonlinearity
    for i, k in enumerate(params.kerr):
        ops.append(
            qml.Kgate(_clip(k, 1) if clip else k, wires=i)
        )

    return ops


class FraudDetection:
    """Variational photonic circuit for fraud detection."""

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        device: qml.Device | None = None,
    ) -> None:
        self.input_params = input_params
        self.layers = list(layers)

        # Use PennyLane's Gaussian device (2 modes)
        self.device = device or qml.device("default.gaussian", wires=2)

        # Build the QNode
        @qml.qnode(self.device, interface="autograd")
        def circuit(*args) -> np.ndarray:
            # Apply input layer (unclipped)
            for op in _apply_layer(self.input_params, clip=False):
                op.apply()
            # Apply subsequent layers (clipped)
            for layer_params in self.layers:
                for op in _apply_layer(layer_params, clip=True):
                    op.apply()
            # Observable: parity on mode 0 as a simple proxy
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def evaluate(self, *args) -> np.ndarray:
        """Run the circuit and return the expectation value."""
        return self.circuit(*args)

    def get_parameters(self) -> List[FraudLayerParameters]:
        """Return the parameters of each layer."""
        return [self.input_params] + self.layers

    @staticmethod
    def from_parameters(
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        device: qml.Device | None = None,
    ) -> "FraudDetection":
        """Convenience constructor."""
        return FraudDetection(input_params, layers, device)

__all__ = ["FraudLayerParameters", "FraudDetection"]
