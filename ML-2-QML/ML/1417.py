"""Enhanced classical fraud detection model with dropout, batch‑norm and training utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn, optim
import torch.nn.functional as F


@dataclass
class FraudLayerParameters:
    """Parameters for a fully‑connected layer in the classical model."""
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


def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
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

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()


class FraudDetectionModel(nn.Module):
    """Full fraud‑detection network with optional dropout and batch‑norm."""

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        *,
        dropout_rate: float = 0.1,
        batch_norm: bool = True,
    ) -> None:
        super().__init__()
        module_list: list[nn.Module] = [_layer_from_params(input_params, clip=False)]
        module_list.extend(_layer_from_params(l, clip=True) for l in layers)

        if batch_norm:
            module_list.append(nn.BatchNorm1d(2))
        if dropout_rate > 0.0:
            module_list.append(nn.Dropout(dropout_rate))

        module_list.append(nn.Linear(2, 1))
        self.network = nn.Sequential(*module_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

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


__all__ = ["FraudLayerParameters", "FraudDetectionModel"]
