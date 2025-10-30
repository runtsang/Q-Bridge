"""Quantum fraud detection circuit with variational layers and training routines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pennylane as qml
import numpy as np


@dataclass
class FraudLayerParameters:
    """Parameters for a variational photonic‑style layer."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


DEVICE = qml.device("default.qubit", wires=2)


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def _apply_layer(params: FraudLayerParameters, clip: bool) -> None:
    """Insert a parametrised entangling block mimicking a photonic layer."""
    # Beam‑splitter analogue: use two‑qubit rotation and CNOT
    qml.RZ(params.bs_phi, wires=0)
    qml.RZ(params.bs_phi, wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RZ(-params.bs_phi, wires=0)
    qml.RZ(-params.bs_phi, wires=1)
    # Single‑qubit rotations encode squeezing and phase
    for i, (phi, r) in enumerate(zip(params.phases, params.squeeze_r)):
        qml.RZ(phi, wires=i)
        qml.RY(r if not clip else _clip(r, 5.0), wires=i)
    # Displacement analogue
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        qml.RZ(phi, wires=i)
        qml.RY(r if not clip else _clip(r, 5.0), wires=i)
    # Kerr non‑linearity analogue
    for i, k in enumerate(params.kerr):
        qml.RZ(k if not clip else _clip(k, 1.0), wires=i)


def build_fraud_detection_circuit(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> qml.QNode:
    """Return a Pennylane QNode implementing the hybrid fraud detection circuit."""

    @qml.qnode(DEVICE, interface="torch")
    def circuit(x: torch.Tensor, params: list[FraudLayerParameters]) -> torch.Tensor:
        # Encode features into initial rotations
        qml.RY(x[0], wires=0)
        qml.RY(x[1], wires=1)
        # First (classical‑like) layer
        _apply_layer(input_params, clip=False)
        # Subsequent variational layers
        for p in params:
            _apply_layer(p, clip=True)
        # Measurement
        return qml.expval(qml.PauliZ(0))

    return circuit


def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Binary cross‑entropy loss for a single sample."""
    return -(target * torch.log(pred + 1e-10) + (1 - target) * torch.log(1 - pred + 1e-10))


def train_qml_model(
    circuit: qml.QNode,
    input_params: FraudLayerParameters,
    layers: list[FraudLayerParameters],
    dataset: Iterable[tuple[np.ndarray, int]],
    epochs: int,
    lr: float = 0.01,
) -> list[FraudLayerParameters]:
    """Gradient‑based training of the variational fraud detection circuit."""
    # Flatten all trainable parameters into a single torch tensor
    param_tensors = [
        torch.tensor([p.bs_theta, p.bs_phi, *p.phases, *p.squeeze_r,
                      *p.squeeze_phi, *p.displacement_r, *p.displacement_phi, *p.kerr],
                     requires_grad=True)
        for p in layers
    ]
    optimizer = torch.optim.Adam(param_tensors, lr=lr)

    for _ in range(epochs):
        for x, y in dataset:
            x_t = torch.tensor(x, dtype=torch.float32)
            preds = circuit(x_t, layers)
            preds = torch.sigmoid(preds)
            loss = loss_fn(preds, torch.tensor(y, dtype=torch.float32))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return layers


def evaluate_qml_model(
    circuit: qml.QNode,
    input_params: FraudLayerParameters,
    layers: list[FraudLayerParameters],
    dataset: Iterable[tuple[np.ndarray, int]],
) -> float:
    """Compute accuracy of the trained QNode on a held‑out set."""
    correct = 0
    total = 0
    for x, y in dataset:
        pred = circuit(torch.tensor(x, dtype=torch.float32), layers).item()
        pred = 1 if torch.sigmoid(torch.tensor(pred)) > 0.5 else 0
        correct += int(pred == y)
        total += 1
    return correct / total


__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_circuit",
    "train_qml_model",
    "evaluate_qml_model",
]
