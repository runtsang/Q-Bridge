"""Pennylane variational fraud detection circuit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import pennylane as qml
import pennylane.numpy as np
import torch
from torch import nn, optim


@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic layer mapped to qubit rotations."""
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


def _layer_circuit(
    params: FraudLayerParameters,
    clip: bool,
    wires: Tuple[int, int],
) -> None:
    """Emit a single layer of rotations and entanglement."""
    theta = params.bs_theta if not clip else _clip(params.bs_theta, 5.0)
    phi = params.bs_phi if not clip else _clip(params.bs_phi, 5.0)
    qml.RZ(theta, wires=wires[0])
    qml.RZ(phi, wires=wires[1])
    qml.CNOT(wires=wires)
    for i, phase in enumerate(params.phases):
        qml.RZ(phase, wires=wires[i])
    for i, (r, ph) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        angle = r if not clip else _clip(r, 5.0)
        qml.RX(angle, wires=wires[i])
    for i, (r, ph) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        angle = r if not clip else _clip(r, 5.0)
        qml.RY(angle, wires=wires[i])
    for i, k in enumerate(params.kerr):
        angle = k if not clip else _clip(k, 1.0)
        qml.RZ(angle, wires=wires[i])


class FraudDetectionQMLModel(nn.Module):
    """Hybrid quantum‑classical fraud detection model with Pennylane."""

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        device: str = "default.qubit",
    ) -> None:
        super().__init__()
        self.layers_params: List[FraudLayerParameters] = [input_params, *layers]
        self.device = device
        self.dev = qml.device(device, wires=2)
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    def _circuit(self, params: torch.Tensor) -> torch.Tensor:
        """Variational circuit producing a single expectation value."""
        # Unpack parameters
        idx = 0
        for layer in self.layers_params:
            # Each layer consumes 18 scalar parameters
            layer_params = FraudLayerParameters(
                bs_theta=params[idx].item(),
                bs_phi=params[idx + 1].item(),
                phases=(params[idx + 2].item(), params[idx + 3].item()),
                squeeze_r=(params[idx + 4].item(), params[idx + 5].item()),
                squeeze_phi=(params[idx + 6].item(), params[idx + 7].item()),
                displacement_r=(params[idx + 8].item(), params[idx + 9].item()),
                displacement_phi=(params[idx + 10].item(), params[idx + 11].item()),
                kerr=(params[idx + 12].item(), params[idx + 13].item()),
            )
            clip = idx > 0  # clip all layers except input
            _layer_circuit(layer_params, clip, wires=(0, 1))
            idx += 18
        return qml.expval(qml.PauliZ(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass – expects a 2‑dim input but ignores it, returning the circuit output."""
        return self.qnode(torch.nn.Parameter(torch.rand(18 * len(self.layers_params))))

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Single training step returning loss."""
        self.optimizer.zero_grad()
        preds = self(x)
        loss = nn.functional.mse_loss(preds, y)
        loss.backward()
        self.optimizer.step()
        return loss

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict using the trained circuit."""
        self.eval()
        with torch.no_grad():
            return self(x)

__all__ = [
    "FraudLayerParameters",
    "FraudDetectionQMLModel",
]
