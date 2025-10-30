"""
Quantum fraud‑detection model using PennyLane.
The circuit is a variational ansatz inspired by the photonic layer
parameters, but implemented with qubit gates for fast simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

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


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


class FraudDetectionModel:
    """
    PennyLane implementation of the fraud‑detection circuit.
    The model encodes each input sample into a 2‑qubit state and
    applies a variational block per photonic layer.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        device_name: str = "default.qubit",
        wires: Sequence[int] | int = 2,
    ) -> None:
        self.input_params = input_params
        self.layers = list(layers)
        self.dev = qml.device(device_name, wires=wires)
        self.qnode = qml.QNode(self._circuit, self.dev)

    def _flatten_params(self) -> np.ndarray:
        """Flatten all layer parameters into a 1‑D array."""
        params = []
        for layer in [self.input_params] + self.layers:
            params.extend(
                [
                    layer.bs_theta,
                    layer.bs_phi,
                    layer.phases[0],
                    layer.phases[1],
                    layer.squeeze_r[0],
                    layer.squeeze_r[1],
                    layer.displacement_r[0],
                    layer.displacement_r[1],
                ]
            )
        return np.array(params, dtype=np.float32)

    def _circuit(self, *params: float) -> float:
        """
        Variational circuit.
        Parameters are unpacked layer‑wise and applied as rotations and entanglers.
        """
        idx = 0
        for _ in self.layers:
            theta = params[idx]
            phi = params[idx + 1]
            p0 = params[idx + 2]
            p1 = params[idx + 3]
            r0 = params[idx + 4]
            r1 = params[idx + 5]
            d0 = params[idx + 6]
            d1 = params[idx + 7]

            # Encode the layer‑specific parameters
            qml.RY(theta, wires=0)
            qml.RZ(phi, wires=1)
            qml.CZ(wires=[0, 1])
            qml.RX(p0, wires=0)
            qml.RX(p1, wires=1)
            qml.RZ(r0, wires=0)
            qml.RZ(r1, wires=1)
            qml.CZ(wires=[0, 1])

            idx += 8

        # Final measurement
        return qml.expval(qml.PauliZ(0))

    def encode_input(self, x: np.ndarray) -> None:
        """Prepare the initial state from the classical input."""
        qml.RY(x[0], wires=0)
        qml.RZ(x[1], wires=1)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Run the circuit for each input sample.
        Args:
            inputs: (batch, 2) array of feature values.
        Returns:
            (batch,) array of expectation values.
        """
        flat_params = self._flatten_params()
        results = []
        for sample in inputs:
            self.encode_input(sample)
            # The qnode receives the flattened parameters
            results.append(self.qnode(*flat_params))
        return np.array(results)

__all__ = ["FraudLayerParameters", "FraudDetectionModel"]
