"""A PyTorch implementation of a fraud detection model inspired by a photonic circuit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn


@dataclass
class FraudLayerParameters:
    """Parameters for one classical layer mirroring a photonic layer."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


class FraudDetectionModel(nn.Module):
    """Modular fraud‑detection neural network inspired by a photonic architecture.

    The network consists of an input layer followed by a stack of
    parameterised layers and a final linear classifier.  Optional
    batch‑normalisation and dropout are available to improve
    generalisation.  Each layer implements a linear map, a tanh
    activation and an affine feature‑wise transformation.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layer_params: Iterable[FraudLayerParameters],
        *,
        dropout: float | None = None,
        batch_norm: bool = False,
        clip: bool = True,
    ) -> None:
        super().__init__()
        layers = [self._layer_from_params(input_params, clip=False)]
        layers.extend(
            self._layer_from_params(p, clip=clip) for p in layer_params
        )
        if batch_norm:
            layers.insert(1, nn.BatchNorm1d(2))
        self.seq = nn.Sequential(*layers)
        self.classifier = nn.Linear(2, 1)
        self.dropout = nn.Dropout(dropout) if dropout else None

    def _layer_from_params(
        self,
        params: FraudLayerParameters,
        *,
        clip: bool,
    ) -> nn.Module:
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
        linear = nn.Linear(2, 2, bias=True)
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

            def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
                out = self.activation(self.linear(x))
                out = out * self.scale + self.shift
                return out

        return Layer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.seq(x)
        if self.dropout:
            x = self.dropout(x)
        return self.classifier(x)


__all__ = ["FraudDetectionModel"]
