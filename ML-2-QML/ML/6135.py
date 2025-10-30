"""Enhanced classical fraud detection with dropout and L2 regularisation."""

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


def _clip(value: float, bound: float) -> float:
    """Clip a scalar to the interval [-bound, bound]."""
    return max(-bound, min(bound, value))


def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    """Create a single linear layer with custom weights and bias derived from params."""
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
    """Classical fraud detection model with optional dropout and L2 regularisation."""

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        dropout_prob: float = 0.0,
        l2_lambda: float = 0.0,
    ) -> None:
        super().__init__()
        modules: list[nn.Module] = [_layer_from_params(input_params, clip=False)]
        modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
        modules.append(nn.Linear(2, 1))
        self.network = nn.Sequential(*modules)
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else nn.Identity()
        self.l2_lambda = l2_lambda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network followed by optional dropout."""
        out = self.network(x)
        out = self.dropout(out)
        return out

    def l2_regularization(self) -> torch.Tensor:
        """Compute L2 penalty over all learnable weights."""
        l2 = torch.tensor(0.0, device="cpu")
        for param in self.parameters():
            l2 += torch.norm(param, 2) ** 2
        return self.l2_lambda * l2

    def loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Binary cross‑entropy loss with optional L2 regularisation."""
        bce = F.binary_cross_entropy_with_logits(predictions.squeeze(), targets)
        return bce + self.l2_regularization()


__all__ = ["FraudLayerParameters", "FraudDetectionModel"]
