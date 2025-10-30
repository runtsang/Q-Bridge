"""Quantum fraud detection circuit using PennyLane.

The module mirrors the classical parameter structure but implements a
variational circuit with entanglement and parameter‑dependent rotations.
The QNode returns a single expectation value that is fed into a small
classical classifier, enabling hybrid training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pennylane as qml
import pennylane.numpy as np
import torch
from torch import nn


@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic layer, identical to the
    classical counterpart so that the same parameter object can be reused.
    """

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


def _apply_layer(wires: Sequence[int], params: FraudLayerParameters, clip: bool) -> None:
    """Apply a photonic‑style layer to the given wires using PennyLane ops."""
    # Beam‑splitter angles -> rotations
    qml.RY(params.bs_theta, wires=wires[0])
    qml.RY(params.bs_phi, wires=wires[1])

    # Phase shifts
    for i, phase in enumerate(params.phases):
        qml.RZ(phase, wires=wires[i])

    # Squeezing -> rotations (as a proxy)
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        qml.RX(r if not clip else _clip(r, 5.0), wires=wires[i])

    # Displacement -> additional rotations
    for i, r in enumerate(params.displacement_r):
        qml.RZ(r, wires=wires[i])

    # Kerr non‑linearity -> phase shift
    for i, k in enumerate(params.kerr):
        qml.RZ(k if not clip else _clip(k, 1.0), wires=wires[i])

    # Entanglement between the two modes
    qml.CNOT(wires=[wires[0], wires[1]])


def build_fraud_detection_qnode(
    dev: qml.Device,
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    *,
    clip: bool = True,
) -> qml.QNode:
    """Return a QNode that evaluates a fraud‑detection circuit.

    The circuit encodes a 2‑dimensional classical input as a pair of
    RX rotations, then applies the photonic layers.  The expectation
    value of PauliZ on wire 0 is returned and later passed to a
    classical classifier.
    """

    @qml.qnode(dev, interface="torch")
    def circuit(x: torch.Tensor) -> torch.Tensor:
        # Encode the classical input
        qml.RX(x[0], wires=0)
        qml.RX(x[1], wires=1)

        # Apply the first (input) layer without clipping
        _apply_layer([0, 1], input_params, clip=False)

        # Apply subsequent layers with clipping
        for layer in layers:
            _apply_layer([0, 1], layer, clip=True)

        # Return a single measurement
        return qml.expval(qml.PauliZ(0))

    return circuit


class FraudQuantumModel(nn.Module):
    """Hybrid quantum‑classical fraud detection model.

    The QNode is wrapped in a PyTorch module so that gradients flow
    through both the quantum circuit and the final classical classifier.
    """

    def __init__(
        self,
        dev: qml.Device,
        input_params: FraudLayerParameters,
        layers: Sequence[FraudLayerParameters],
        *,
        clip: bool = True,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        self.circuit = build_fraud_detection_qnode(dev, input_params, layers, clip=clip)
        self.classifier = nn.Linear(1, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The circuit returns a scalar expectation value
        out = self.circuit(x)
        out = self.classifier(out)
        return out


__all__ = ["FraudLayerParameters", "FraudQuantumModel", "build_fraud_detection_qnode"]
