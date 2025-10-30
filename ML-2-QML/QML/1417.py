"""Pennylane variational circuit for fraud detection with a hybrid classical‑quantum forward pass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pennylane as qml
import torch
from torch import nn, optim
import torch.nn.functional as F


@dataclass
class FraudLayerParameters:
    """Parameters for a single photonic‑style layer (used only for compatibility)."""
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


def _apply_layer(
    params: FraudLayerParameters, *, clip: bool
) -> list[qml.operation.Operator]:
    ops: list[qml.operation.Operator] = []
    ops.append(qml.BSgate(params.bs_theta, params.bs_phi))
    for phase in params.phases:
        ops.append(qml.Rgate(phase))
    for r, phi in zip(params.squeeze_r, params.squeeze_phi):
        ops.append(qml.Sgate(r if not clip else _clip(r, 5), phi))
    ops.append(qml.BSgate(params.bs_theta, params.bs_phi))
    for phase in params.phases:
        ops.append(qml.Rgate(phase))
    for r, phi in zip(params.displacement_r, params.displacement_phi):
        ops.append(qml.Dgate(r if not clip else _clip(r, 5), phi))
    for k in params.kerr:
        ops.append(qml.Kgate(k if not clip else _clip(k, 1)))
    return ops


class FraudDetectionQuantumModel(nn.Module):
    """
    Hybrid classical‑quantum fraud‑detection model.
    The quantum part is a Pennylane variational circuit with 2 modes.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        *,
        n_layers: int = 2,
        device_name: str = "default.qubit",
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.device = qml.device(device_name, wires=2)
        self.params = [input_params] + list(layers)

        # Classical head: simple linear layer
        self.classical_head = nn.Linear(2, 1)

        # Prepare trainable parameters for the variational circuit
        self.var_params = nn.Parameter(
            torch.randn(n_layers * 3, dtype=torch.float64)  # 3 params per layer
        )

        # QNode
        @qml.qnode(self.device, interface="torch")
        def circuit(x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
            # Encode input
            qml.QubitStateVector(x, wires=range(2))
            # Variational layers
            for i in range(n_layers):
                theta, phi, chi = params[i * 3 : (i + 1) * 3]
                qml.RX(theta, wires=0)
                qml.RY(phi, wires=1)
                qml.CZ(wires=[0, 1])
                qml.RZ(chi, wires=0)
            # Measurement
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantum forward
        q_out = self.circuit(x, self.var_params)
        # Classical head
        return self.classical_head(q_out.unsqueeze(1))

    def loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(logits.squeeze(), targets.float())

    def train_step(
        self,
        optimizer: optim.Optimizer,
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        self.train()
        optimizer.zero_grad()
        logits = self(batch[0])
        loss = self.loss(logits, batch[1])
        loss.backward()
        optimizer.step()
        return loss.detach()

    def fit(
        self,
        data_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
        epochs: int = 10,
        lr: float = 1e-3,
    ) -> None:
        optimizer = optim.Adam(self.parameters(), lr=lr)
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in data_loader:
                loss = self.train_step(optimizer, batch)
                epoch_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs} – loss: {epoch_loss / len(data_loader):.4f}")


__all__ = ["FraudLayerParameters", "FraudDetectionQuantumModel"]
