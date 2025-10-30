"""Variational quantum fraud detection model using PennyLane."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import pennylane as qml
import numpy as np
import torch


@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic layer (used as a template for a qubit circuit)."""

    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]
    dropout_rate: float = 0.0


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def _apply_layer(wires: Sequence[int], params: FraudLayerParameters, *, clip: bool) -> None:
    # Entangling layer (mimicking a beam splitter)
    qml.CNOT(wires=[wires[0], wires[1]])
    # Phase rotations
    for i, phase in enumerate(params.phases):
        qml.RX(phase, wires=wires[i])
    # Squeezing analogue (RZ rotations)
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        qml.RZ(_clip(r, 5) if clip else r, wires=wires[i])
    # Second entangling
    qml.CNOT(wires=[wires[0], wires[1]])
    # Additional phases
    for i, phase in enumerate(params.phases):
        qml.RX(phase, wires=wires[i])
    # Displacement analogue (RY rotations)
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        qml.RY(_clip(r, 5) if clip else r, wires=wires[i])
    # Kerr analogue (RZ rotations)
    for i, k in enumerate(params.kerr):
        qml.RZ(_clip(k, 1) if clip else k, wires=wires[i])


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> qml.QNode:
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs: torch.Tensor):
        # Encode classical inputs as rotation angles
        qml.RX(inputs[0], wires=0)
        qml.RX(inputs[1], wires=1)
        _apply_layer(dev.wires, input_params, clip=False)
        for layer in layers:
            _apply_layer(dev.wires, layer, clip=True)
        # Readout (PauliZ expectation on first qubit)
        return qml.expval(qml.PauliZ(0))

    return circuit


class FraudDetectionHybrid:
    """Quantum‑variational fraud detection model with training utilities."""

    def __init__(self, input_params: FraudLayerParameters, layers: Iterable[FraudLayerParameters]):
        self.circuit = build_fraud_detection_program(input_params, layers)
        self.params = self.circuit.trainable_params
        self.optimizer = qml.AdamOptimizer(stepsize=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.circuit(x)

    def loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return torch.mean((preds - targets) ** 2)

    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        inputs, targets = batch

        def cost_fn(p):
            preds = self.circuit(inputs)
            return self.loss(preds, targets)

        self.params, _ = self.optimizer.step_and_cost(cost_fn, self.params)
        return cost_fn(self.params)


__all__ = ["FraudLayerParameters", "build_fraud_detection_program", "FraudDetectionHybrid"]
