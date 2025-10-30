"""
FraudDetectionModel – Classical PyTorch implementation.

The class builds a deep feed‑forward network that mirrors the
photonic layout from the seed.  Each layer contains a linear
transformation, a tanh activation, and a learnable scale‑shift
post‑processing step.  The network can be trained end‑to‑end with
binary cross‑entropy loss.  A helper class `ScaleShiftLayer` is
provided to expose the shift/scale as buffers so they are not
optimised by default.

Author: gpt-oss-20b
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn


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


class ScaleShiftLayer(nn.Module):
    """
    Learnable scale and shift applied after the non‑linearity.
    The buffers are not updated by optimisers unless explicitly
    added to the parameter list.
    """
    def __init__(self, init_scale: Sequence[float] | None = None,
                 init_shift: Sequence[float] | None = None) -> None:
        super().__init__()
        scale = torch.tensor(init_scale if init_scale else [1.0, 1.0],
                             dtype=torch.float32)
        shift = torch.tensor(init_shift if init_shift else [0.0, 0.0],
                             dtype=torch.float32)
        self.register_buffer("scale", scale)
        self.register_buffer("shift", shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.shift


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    weight = torch.tensor(
        [[params.bs_theta, params.bs_phi],
         [params.squeeze_r[0], params.squeeze_r[1]]],
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
    """
    Classical fraud‑detection network that mirrors the photonic
    circuit from the seed.  The constructor accepts a list of
    `FraudLayerParameters` to build the corresponding layers.
    """
    def __init__(self,
                 input_params: FraudLayerParameters,
                 layers: Iterable[FraudLayerParameters]) -> None:
        super().__init__()
        modules = [_layer_from_params(input_params, clip=False)]
        modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
        modules.append(nn.Linear(2, 1))  # final binary output
        self.model = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(x)
        return torch.sigmoid(logits)

__all__ = ["FraudLayerParameters", "FraudDetectionModel"]
