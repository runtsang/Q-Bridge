"""
FraudDetectionModel – Quantum implementation using PennyLane.

The circuit implements a two‑mode photonic‑like structure with
parameterised beamsplitters, squeezers, displacements, and Kerr
non‑linearities mapped to elementary qubit gates.  The model
returns the expectation value of Pauli‑Z on the first qubit,
matching the classical output dimension.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pennylane as qml
import torch
from torch import nn


@dataclass
class LayerParameters:
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


class FraudDetectionModel(nn.Module):
    """
    Quantum fraud‑detection model built with PennyLane.

    Parameters
    ----------
    input_params : LayerParameters
        Parameters for the first (input) layer.
    hidden_params : Iterable[LayerParameters]
        Parameters for subsequent hidden layers.
    """

    def __init__(
        self,
        input_params: LayerParameters,
        hidden_params: Iterable[LayerParameters],
    ) -> None:
        super().__init__()
        self.input_params = input_params
        self.hidden_params = list(hidden_params)

        self.dev = qml.device("default.qubit", wires=2)
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")

    def _circuit(self, x: torch.Tensor) -> torch.Tensor:
        """Builds the layered photonic‑like circuit."""
        # Input encoding
        qml.RX(x[0], wires=0)
        qml.RY(x[1], wires=1)

        # Apply input layer
        self._apply_layer(self.input_params, clip=False)

        # Hidden layers
        for params in self.hidden_params:
            self._apply_layer(params, clip=True)

        # Measurement
        return qml.expval(qml.PauliZ(0))

    def _apply_layer(self, params: LayerParameters, *, clip: bool) -> None:
        """Map photonic operations to qubit gates."""
        # Beamsplitter → RXX (approximation)
        theta = params.bs_theta
        phi = params.bs_phi
        qml.RXX(theta, wires=[0, 1])
        qml.RYY(phi, wires=[0, 1])

        # Phase shifters
        for i, phase in enumerate(params.phases):
            qml.RZ(phase, wires=i)

        # Squeezing → RZ + RX (approximation)
        for i, (r, ph) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
            r_clipped = _clip(r, 5.0) if clip else r
            qml.RZ(r_clipped, wires=i)
            qml.RX(ph, wires=i)

        # Displacements → RX + RZ
        for i, (r, ph) in enumerate(zip(params.displacement_r, params.displacement_phi)):
            r_clipped = _clip(r, 5.0) if clip else r
            qml.RX(r_clipped, wires=i)
            qml.RZ(ph, wires=i)

        # Kerr non‑linearity → RZ (approximation)
        for i, k in enumerate(params.kerr):
            k_clipped = _clip(k, 1.0) if clip else k
            qml.RZ(k_clipped, wires=i)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.qnode(x)

__all__ = ["LayerParameters", "FraudDetectionModel"]
