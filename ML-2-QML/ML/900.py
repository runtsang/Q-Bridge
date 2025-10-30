"""Extended classical fraud detection model with batch‑norm, dropout and training utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn
import torch.nn.functional as F


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
    dropout: float = 0.0  # new attribute for regularisation


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
            self.batchnorm = nn.BatchNorm1d(2)
            self.dropout = nn.Dropout(p=params.dropout)
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            x = self.linear(inputs)
            x = self.activation(x)
            x = self.batchnorm(x)
            x = self.dropout(x)
            x = x * self.scale + self.shift
            return x

    return Layer()


class FraudDetectionHybrid(nn.Module):
    """A PyTorch model that mirrors the layered photonic architecture with added regularisation."""

    def __init__(self, input_params: FraudLayerParameters, layers: Iterable[FraudLayerParameters]) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _layer_from_params(input_params, clip=False),
            *(_layer_from_params(layer, clip=True) for layer in layers),
            nn.Linear(2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)

    def train_step(self, optimizer, criterion, data: torch.Tensor, target: torch.Tensor) -> float:
        """One optimisation step."""
        self.train()
        optimizer.zero_grad()
        output = self(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        return loss.item()

    def evaluate(self, data: torch.Tensor, target: torch.Tensor) -> float:
        """Evaluation using MSE."""
        self.eval()
        with torch.no_grad():
            output = self(data)
            loss = F.mse_loss(output, target)
        return loss.item()


__all__ = ["FraudLayerParameters", "FraudDetectionHybrid"]
