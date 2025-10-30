"""Quantum fraud detection model using PennyLane."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pennylane as qml
import torch


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
    depth: int = 1  # future‑proofing for deeper ansatzes


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def _apply_layer(params: FraudLayerParameters, clip: bool) -> None:
    """Photonic‑inspired layer built from PennyLane primitives."""
    # Beam‑splitter analogues via rotations
    qml.RZ(params.bs_phi, wires=0)
    qml.RZ(params.bs_phi, wires=1)

    # Phase gates
    for i, phase in enumerate(params.phases):
        qml.RZ(phase, wires=i)

    # Squeezing analogues
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        qml.RX(r if not clip else _clip(r, 5), wires=i)
        qml.RZ(phi, wires=i)

    # Displacement analogues
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        qml.RY(r if not clip else _clip(r, 5), wires=i)
        qml.RZ(phi, wires=i)

    # Kerr analogues via ZZ interactions
    for i, k in enumerate(params.kerr):
        qml.RZZ(k if not clip else _clip(k, 1), wires=[i, (i + 1) % 2])

    # Entanglement
    qml.CNOT(wires=[0, 1])


class FraudDetectionModel:
    """Quantum fraud detection model with a variational circuit and classical readout."""

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        *,
        device: str = "default.qubit",
        wires: int = 2,
    ) -> None:
        self.device = qml.device(device, wires=wires)
        self.input_params = input_params
        self.layers = list(layers)

        @qml.qnode(self.device, interface="torch")
        def circuit(inputs: torch.Tensor) -> torch.Tensor:
            # Encode classical inputs as RX rotations
            qml.RX(inputs[0], wires=0)
            qml.RX(inputs[1], wires=1)

            # Apply photonic‑inspired layers
            _apply_layer(self.input_params, clip=False)
            for layer in self.layers:
                _apply_layer(layer, clip=True)

            # Classical post‑processing: expectation value of Pauli‑Z on qubit 0
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.circuit(x)

    @classmethod
    def build_from_params(
        cls,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        *,
        device: str = "default.qubit",
        wires: int = 2,
    ) -> "FraudDetectionModel":
        return cls(input_params, layers, device=device, wires=wires)


__all__ = ["FraudLayerParameters", "FraudDetectionModel"]
