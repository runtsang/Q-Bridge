"""Enhanced classical fraud‑detection model with residual connections and dropout.

The module builds on the original layer definition but adds a residual
shortcut, dropout after the activation, and a lightweight training
interface.  It remains fully compatible with the original
FraudLayerParameters dataclass and can be used interchangeably in
experiments that previously relied on ``build_fraud_detection_program``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn
from torch.nn import functional as F


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


def _clip(value: float, bound: float) -> float:
    """Clamp a scalar to the interval [-bound, bound]."""
    return max(-bound, min(bound, value))


def _layer_from_params(params: FraudLayerParameters, *, clip: bool, dropout: float = 0.0) -> nn.Module:
    """Create a single linear‑activation‑scale block from parameters."""
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
            self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            x = self.linear(inputs)
            x = self.activation(x)
            x = self.drop(x)
            x = x * self.scale + self.shift
            return x

    return Layer()


class FraudDetectionModel(nn.Module):
    """Residual‑dropout neural network that emulates the photonic fraud circuit."""
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        modules = [_layer_from_params(input_params, clip=False, dropout=dropout)]
        modules.extend(
            _layer_from_params(layer, clip=True, dropout=dropout) for layer in layers
        )
        # Residual shortcut from input to final layer
        self.residual = nn.Linear(2, 2)
        self.final = nn.Linear(2, 1)
        self.sequence = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with a residual connection."""
        out = self.sequence(x)
        res = self.residual(x)
        out = out + res
        return self.final(out)

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """Perform a single SGD step and return the loss."""
        self.train()
        optimizer.zero_grad()
        preds = self(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        return loss.item()


__all__ = ["FraudLayerParameters", "FraudDetectionModel", "_layer_from_params"]
