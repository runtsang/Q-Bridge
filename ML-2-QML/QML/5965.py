"""Quantum‑enhanced fraud detection using a PennyLane variational circuit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pennylane as qml
import numpy as np
import torch


@dataclass
class FraudLayerParameters:
    """Parameters describing a single variational photonic layer."""
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


def _apply_layer(circuit, params: FraudLayerParameters, *, clip: bool) -> None:
    """Append a photonic‑style variational block to a PennyLane circuit."""
    # Beam‑splitter style rotations
    qml.RX(params.bs_theta, wires=0)
    qml.RY(params.bs_phi, wires=1)
    # Phase shifts
    for i, phase in enumerate(params.phases):
        qml.RZ(phase, wires=i)
    # Squeezing‑like rotations (approximated with RZ)
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        qml.RZ(_clip(r, 5.0) if clip else r, wires=i)
    # Displacement‑like rotations
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        qml.RX(_clip(r, 5.0) if clip else r, wires=i)
    # Kerr‑like nonlinearity (approximated with RZ)
    for i, k in enumerate(params.kerr):
        qml.RZ(_clip(k, 1.0) if clip else k, wires=i)


def build_fraud_detection_qnode(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    device: str | qml.Device = "default.qubit",
) -> qml.QNode:
    """Create a PennyLane QNode that encodes the fraud‑detection parameters."""
    dev = qml.device(device, wires=2)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs: torch.Tensor, params: list[FraudLayerParameters]) -> torch.Tensor:
        # Encode classical features as initial rotations
        qml.RX(inputs[0], wires=0)
        qml.RY(inputs[1], wires=1)

        _apply_layer(circuit, input_params, clip=False)
        for layer in params:
            _apply_layer(circuit, layer, clip=True)

        # Output expectation value of Z on first qubit
        return qml.expval(qml.PauliZ(0))

    return circuit


class FraudDetectionModel:
    """Quantum‑only fraud‑detection wrapper using a PennyLane variational circuit."""
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        device: str | qml.Device = "default.qubit",
    ) -> None:
        self.circuit = build_fraud_detection_qnode(input_params, layers, device=device)

    def forward(self, x: np.ndarray | torch.Tensor) -> float:
        """Run the QNode with the given classical input."""
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        return self.circuit(x, list(self.circuit.parameters))[0].item()

    def parameters(self) -> list[torch.Tensor]:
        """Return the trainable parameters of the circuit."""
        return list(self.circuit.parameters)


__all__ = ["FraudLayerParameters", "build_fraud_detection_qnode", "FraudDetectionModel"]
