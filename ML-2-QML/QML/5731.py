"""Quantum‑enhanced fraud detection circuit using PennyLane."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pennylane as qml
import torch


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
    """Clamp values to a safe range for numerical stability."""
    return max(-bound, min(bound, value))


class FraudDetectionEnhanced:
    """Encapsulates a variational photonic circuit for fraud detection."""

    def __init__(
        self,
        device: qml.Device,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        *,
        clip: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        device
            PennyLane device on which to execute the circuit.
        input_params
            Parameters for the first (input) layer.
        layers
            Iterable of parameters for subsequent layers.
        clip
            Whether to clip parameter values for numerical stability.
        """
        self.device = device
        self.input_params = input_params
        self.layers = list(layers)
        self.clip = clip
        self.qnode = self._build_qnode()

    def _build_qnode(self) -> qml.QNode:
        """Construct a PennyLane QNode with parameter‑shift gradients."""

        @qml.qnode(self.device, interface="torch", diff_method="parameter-shift")
        def circuit(inputs: torch.Tensor) -> torch.Tensor:
            # Encode inputs as displacements
            qml.Displacement(inputs[0], 0.0, wires=0)
            qml.Displacement(inputs[1], 0.0, wires=1)

            # Input layer
            self._apply_layer(self.input_params, clip=False)

            # Subsequent layers
            for layer in self.layers:
                self._apply_layer(layer, clip=self.clip)

            # Measure photon number on both modes
            return qml.expval(qml.NumberOperator(wires=0)), qml.expval(qml.NumberOperator(wires=1))

        return circuit

    def _apply_layer(
        self,
        params: FraudLayerParameters,
        *,
        clip: bool,
    ) -> None:
        """Append a photonic layer to the current circuit context."""
        # Beam splitter
        qml.BSgate(params.bs_theta, params.bs_phi, wires=[0, 1])

        # Phase shifts
        for i, phase in enumerate(params.phases):
            qml.Rgate(phase, wires=i)

        # Squeezing
        for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
            qml.Sgate(r if not clip else _clip(r, 5.0), phi, wires=i)

        # Second beam splitter
        qml.BSgate(params.bs_theta, params.bs_phi, wires=[0, 1])

        # Phase shifts again
        for i, phase in enumerate(params.phases):
            qml.Rgate(phase, wires=i)

        # Displacement
        for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
            qml.Dgate(r if not clip else _clip(r, 5.0), phi, wires=i)

        # Kerr nonlinearity
        for i, k in enumerate(params.kerr):
            qml.Kerr(k if not clip else _clip(k, 1.0), wires=i)

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        """Evaluate the circuit on the given inputs."""
        return self.qnode(inputs)

    def gradient(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the gradient of the circuit output w.r.t. the trainable parameters."""
        return qml.grad(self.qnode)(inputs)

__all__ = ["FraudDetectionEnhanced", "FraudLayerParameters"]
