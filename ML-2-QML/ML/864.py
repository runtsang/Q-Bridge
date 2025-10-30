"""Enhanced classical fraud‑detection model with feature selection, dropout, and hybrid loss."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn
import torch.nn.functional as F

# For quantum integration
import pennylane as qml
import pennylane.numpy as np


@dataclass
class FraudLayerParameters:
    """Parameters describing a fully connected layer in the classical model."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]
    dropout: float = 0.0  # new field


def _clip(value: float, bound: float) -> float:
    """Clamp a scalar to a symmetric interval."""
    return max(-bound, min(bound, value))


def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    """Return a single linear‑activation‑dropout‑scale‑shift module."""
    weight = torch.tensor(
        [
            [params.bs_theta, params.bs_phi],
            [params.squeeze_r[0], params.squeeze_r[1]],
        ],
        dtype=torch.float32,
    )
    bias = torch.tensor(params.phases, dtype=torch.float32)
    if clip:
        weight = weight.clamp(-5.0, 5.0)
        bias = bias.clamp(-5.0, 5.0)
    linear = nn.Linear(2, 2)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)
    activation = nn.Tanh()
    scale = torch.tensor(params.displacement_r, dtype=torch.float32)
    shift = torch.tensor(params.displacement_phi, dtype=torch.float32)
    dropout = nn.Dropout(params.dropout)

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.dropout = dropout
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            outputs = self.activation(self.linear(inputs))
            outputs = self.dropout(outputs)
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Create a sequential PyTorch model mirroring the layered structure with dropout."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


class FraudDetectionEnhancedModel(nn.Module):
    """Hybrid fraud‑detection model that combines a classical network with a variational quantum circuit."""
    def __init__(
        self,
        class_params: FraudLayerParameters,
        class_layers: Sequence[FraudLayerParameters],
        quantum_params: FraudLayerParameters,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.device = device
        self.classical = build_fraud_detection_program(class_params, class_layers).to(device)
        self.quantum_params = quantum_params
        self.qnode = self._build_qnode(quantum_params)

    def _build_qnode(self, params: FraudLayerParameters) -> qml.QNode:
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="torch")
        def circuit(x: torch.Tensor) -> torch.Tensor:
            # Encode features
            qml.RY(x[0], wires=0)
            qml.RY(x[1], wires=1)
            # Variational part
            qml.RZ(params.bs_theta, wires=0)
            qml.RZ(params.bs_phi, wires=1)
            qml.CZ(wires=[0, 1])
            # Measurement
            return qml.expval(qml.PauliZ(0))

        return circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return classical output and quantum output."""
        x = x.to(self.device)
        class_out = self.classical(x).squeeze(-1)
        quantum_out = self.qnode(x).squeeze(-1)
        return class_out, quantum_out


def hybrid_loss(
    class_out: torch.Tensor,
    quantum_out: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.5,
) -> torch.Tensor:
    """Weighted BCE loss between target and a convex combination of classical and quantum outputs."""
    combined = alpha * class_out + (1 - alpha) * quantum_out
    return F.binary_cross_entropy_with_logits(combined, target)


def train(
    model: nn.Module,
    data_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    epochs: int,
    lr: float = 1e-3,
    alpha: float = 0.5,
    device: torch.device = torch.device("cpu"),
) -> None:
    """Simple training loop for the hybrid model."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        epoch_loss = 0.0
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            class_out, quantum_out = model(inputs)
            loss = hybrid_loss(class_out, quantum_out, targets, alpha)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs} – loss: {epoch_loss / len(data_loader):.4f}")


__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "FraudDetectionEnhancedModel",
    "hybrid_loss",
    "train",
]
