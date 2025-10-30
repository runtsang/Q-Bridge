"""Quantum implementation of the fraud‑detection hybrid model using PennyLane."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pennylane as qml
import pennylane.numpy as np
import torch
from torch import Tensor


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


def _apply_layer(params: FraudLayerParameters, *, clip: bool) -> None:
    """Apply a photonic‑style layer using qubit gates."""
    qml.CRX(params.bs_theta, wires=0)
    qml.CRX(params.bs_phi, wires=1)

    for i, phase in enumerate(params.phases):
        qml.RZ(phase, wires=i)

    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        qml.RY(r if not clip else _clip(r, 5.0), wires=i)
        qml.RZ(phi, wires=i)

    qml.CRX(params.bs_theta, wires=0)
    qml.CRX(params.bs_phi, wires=1)

    for i, phase in enumerate(params.phases):
        qml.RZ(phase, wires=i)

    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        qml.RX(r if not clip else _clip(r, 5.0), wires=i)
        qml.RZ(phi, wires=i)

    for i, k in enumerate(params.kerr):
        qml.RZ(k if not clip else _clip(k, 1.0), wires=i)


def build_fraud_detection_qnode(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    dev: qml.Device,
) -> qml.QNode:
    """Return a QNode that mirrors the photonic fraud‑detection circuit."""

    @qml.qnode(dev, interface="torch")
    def circuit(inputs: np.ndarray, *flat_params: float) -> np.ndarray:
        idx = 0

        def unpack() -> FraudLayerParameters:
            nonlocal idx
            params = FraudLayerParameters(
                bs_theta=flat_params[idx],
                bs_phi=flat_params[idx + 1],
                phases=(flat_params[idx + 2], flat_params[idx + 3]),
                squeeze_r=(flat_params[idx + 4], flat_params[idx + 5]),
                squeeze_phi=(flat_params[idx + 6], flat_params[idx + 7]),
                displacement_r=(flat_params[idx + 8], flat_params[idx + 9]),
                displacement_phi=(flat_params[idx + 10], flat_params[idx + 11]),
                kerr=(flat_params[idx + 12], flat_params[idx + 13]),
            )
            idx += 14
            return params

        # Input layer
        _apply_layer(unpack(), clip=False)

        # Subsequent layers
        for _ in range(len(layers)):
            _apply_layer(unpack(), clip=True)

        return qml.expval(qml.PauliZ(0))

    return circuit


class FraudDetectionHybridModel:
    """Quantum wrapper that exposes a predict method compatible with the classical model."""

    def __init__(self, device: str = "default.qubit", wires: int = 2) -> None:
        self.dev = qml.device(device, wires=wires)
        self.circuit: qml.QNode | None = None
        self.input_params: FraudLayerParameters | None = None
        self.layers: Sequence[FraudLayerParameters] | None = None

    def build(
        self,
        input_params: FraudLayerParameters,
        layers: Sequence[FraudLayerParameters],
    ) -> None:
        """Instantiate the quantum circuit and store the QNode."""
        self.input_params = input_params
        self.layers = layers
        self.circuit = build_fraud_detection_qnode(input_params, layers, self.dev)

    def predict(self, inputs: Tensor) -> Tensor:
        if self.circuit is None:
            raise RuntimeError("Quantum circuit not built. Call build() first.")
        flat_params: list[float] = []

        for params in [self.input_params] + list(self.layers):
            flat_params.extend(
                [
                    params.bs_theta,
                    params.bs_phi,
                    *params.phases,
                    *params.squeeze_r,
                    *params.squeeze_phi,
                    *params.displacement_r,
                    *params.displacement_phi,
                    *params.kerr,
                ]
            )
        return self.circuit(inputs, *flat_params)


__all__ = [
    "FraudLayerParameters",
    "FraudDetectionHybridModel",
    "build_fraud_detection_qnode",
]
