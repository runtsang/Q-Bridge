"""Hybrid photonic fraud detection using PennyLane."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import pennylane as qml
import pennylane.numpy as pnp


@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic layer."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def _apply_layer(
    wires: Sequence[int], params: FraudLayerParameters, *, clip: bool
) -> None:
    """Append a photonic layer to the current circuit."""
    qml.BSgate(params.bs_theta, params.bs_phi, wires=wires)
    for i, phase in enumerate(params.phases):
        qml.Rgate(phase, wires=wires[i])
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        qml.Sgate(r if not clip else _clip(r, 5.0), phi, wires=wires[i])
    qml.BSgate(params.bs_theta, params.bs_phi, wires=wires)
    for i, phase in enumerate(params.phases):
        qml.Rgate(phase, wires=wires[i])
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        qml.Dgate(r if not clip else _clip(r, 5.0), phi, wires=wires[i])
    for i, k in enumerate(params.kerr):
        qml.Kgate(k if not clip else _clip(k, 1.0), wires=wires[i])


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> qml.QNode:
    """
    Create a PennyLane QNode that implements the hybrid fraud detection circuit.
    The QNode takes a 2‑dimensional classical input vector and returns the expectation
    value of PauliZ on wire 0, suitable for use in a hybrid training loop.
    """
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs: Sequence[float]) -> pnp.ndarray:
        # Encode classical input as displacement on each mode
        qml.Dgate(inputs[0], 0.0, wires=0)
        qml.Dgate(inputs[1], 0.0, wires=1)

        # Apply the first photonic layer
        _apply_layer([0, 1], input_params, clip=False)

        # Apply subsequent layers
        for layer in layers:
            _apply_layer([0, 1], layer, clip=True)

        # Measurement
        return qml.expval(qml.PauliZ(0))

    return circuit


class FraudDetectionQuantumModel:
    """Convenience wrapper for the PennyLane QNode."""
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
    ) -> None:
        self.qnode = build_fraud_detection_program(input_params, layers)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass that accepts a torch tensor of shape (batch, 2)."""
        return torch.stack([self.qnode(x[i].tolist()) for i in range(x.shape[0])])

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        """Compute gradients via parameter‑shift rule."""
        return torch.autograd.grad(
            outputs=self.__call__(x),
            inputs=x,
            grad_outputs=torch.ones_like(self.__call__(x)),
            retain_graph=True,
            create_graph=True,
        )[0]


__all__ = ["FraudLayerParameters", "build_fraud_detection_program", "FraudDetectionQuantumModel"]
