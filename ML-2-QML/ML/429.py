"""FraudDetectionNetwork: classical model with enhanced regularisation and dropout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

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
            self.batchnorm = nn.BatchNorm1d(2)
            self.dropout = nn.Dropout(p=0.2)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            x = self.linear(inputs)
            x = self.activation(x)
            x = self.batchnorm(x)
            x = self.dropout(x)
            x = x * self.scale + self.shift
            return x

    return Layer()


class FraudDetectionNetwork(nn.Module):
    """Enhanced classical fraud‑detection network.

    The architecture mirrors the original photonic‑like design but adds
    batch‑normalisation, dropout and an optional weight‑decay term in the
    loss function.  It can be trained with any optimiser; the ``loss`` method
    incorporates L2 regularisation for convenience.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        *,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__()
        self.weight_decay = weight_decay
        modules = [_layer_from_params(input_params, clip=False)]
        modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
        modules.append(nn.Linear(2, 1))
        self.model = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass producing a single output value per sample."""
        return self.model(x)

    def loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Binary‑cross‑entropy loss with optional L2 weight‑decay."""
        bce = F.binary_cross_entropy_with_logits(logits.squeeze(-1), targets)
        if self.weight_decay > 0:
            l2 = sum(p.pow(2).sum() for p in self.parameters())
            return bce + self.weight_decay * l2
        return bce
