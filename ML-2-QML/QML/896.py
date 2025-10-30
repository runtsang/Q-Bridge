"""Quantum‑classical fraud detection model using PennyLane."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, List

import pennylane as qml
import torch
import torch.nn as nn
import numpy as np


@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic layer adapted for qubit gates."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]
    use_feature_map: bool = False  # new flag to enable classical feature embedding


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


class FraudDetectionModel(nn.Module):
    """Hybrid quantum‑classical fraud detection model built with PennyLane."""

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        *,
        device: str = "default.qubit",
        wires: int = 2,
        num_shots: int = 8192,
    ) -> None:
        super().__init__()
        self.wires = wires
        self.dev = qml.device(device, wires=self.wires, shots=num_shots)
        self.input_params = input_params
        self.layers = list(layers)

        # Build the variational circuit
        @qml.qnode(self.dev, interface="torch")
        def circuit(x: torch.Tensor) -> torch.Tensor:
            # Optional classical feature map
            if self.input_params.use_feature_map:
                for i, val in enumerate(x):
                    qml.RY(val, wires=i)
            # Apply photonic‑style gates mapped to qubit operations
            self._apply_layer(x, self.input_params, clip=False)
            for layer in self.layers:
                self._apply_layer(x, layer, clip=True)
            # Measurement: expectation of PauliZ on wire 0
            return qml.expval(qml.PauliZ(0))

        self.qnode = circuit
        # Classical linear classifier to map quantum expectation to probability
        self.classifier = nn.Linear(1, 1)
        self.sigmoid = nn.Sigmoid()

    def _apply_layer(self, x: torch.Tensor, params: FraudLayerParameters, *, clip: bool) -> None:
        """Map photonic layer parameters to qubit gates."""
        # Beam splitter analog: rotation around Y
        theta = params.bs_theta if not clip else _clip(params.bs_theta, 5.0)
        phi = params.bs_phi if not clip else _clip(params.bs_phi, 5.0)
        qml.RY(theta, wires=0)
        qml.RZ(phi, wires=0)
        # Phase gates
        for i, phase in enumerate(params.phases):
            qml.RZ(phase, wires=i)
        # Squeezing analog: rotations around X
        for i, (r, ph) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
            r_val = r if not clip else _clip(r, 5.0)
            qml.RX(r_val, wires=i)
        # Displacement analog: rotations around Y
        for i, (r, ph) in enumerate(zip(params.displacement_r, params.displacement_phi)):
            r_val = r if not clip else _clip(r, 5.0)
            qml.RY(r_val, wires=i)
        # Kerr analog: Z rotations
        for i, k in enumerate(params.kerr):
            k_val = k if not clip else _clip(k, 1.0)
            qml.RZ(k_val, wires=i)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute fraud probability for a batch of 2‑dimensional inputs."""
        # Ensure input shape (batch, 2)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        quantum_out = self.qnode(x)
        logits = self.classifier(quantum_out)
        return self.sigmoid(logits)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return binary predictions."""
        probs = self.forward(x)
        return (probs > 0.5).float()

    def gradients(self, x: torch.Tensor) -> torch.Tensor:
        """Return gradients of the output w.r.t. the input features."""
        x.requires_grad_(True)
        probs = self.forward(x)
        probs.sum().backward()
        return x.grad

    def train_step(
        self,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """Single training step."""
        optimizer.zero_grad()
        preds = self.forward(x_batch)
        loss = loss_fn(preds, y_batch)
        loss.backward()
        optimizer.step()
        return loss.item()

    def to_qubit_device(self, device: str) -> None:
        """Re‑initialize the circuit on a new device."""
        self.dev = qml.device(device, wires=self.wires, shots=self.dev.shots)
        # Recreate the qnode with the new device
        @qml.qnode(self.dev, interface="torch")
        def circuit(x: torch.Tensor) -> torch.Tensor:
            if self.input_params.use_feature_map:
                for i, val in enumerate(x):
                    qml.RY(val, wires=i)
            self._apply_layer(x, self.input_params, clip=False)
            for layer in self.layers:
                self._apply_layer(x, layer, clip=True)
            return qml.expval(qml.PauliZ(0))
        self.qnode = circuit

__all__ = ["FraudLayerParameters", "FraudDetectionModel"]
