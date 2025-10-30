"""
A hybrid classical fraud‑detection model that extends the original
photonic analogue by attaching a deep feed‑forward head.  The model
mirrors the layer construction of the quantum circuit but replaces
the final linear output with a lightweight neural network that
learns a non‑linear decision boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


@dataclass
class FraudLayerParameters:
    """
    Parameters for a single photonic layer.  These are re‑used for the
    classical analogue.  They are kept identical to the quantum seed
    to facilitate a one‑to‑one mapping between the two models.
    """
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


def _clip(value: float, bound: float) -> float:
    """Utility to keep parameters bounded during training."""
    return max(-bound, min(bound, value))


def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    """
    Build a single classical layer that mimics the photonic construction.
    The layer contains a 2×2 linear map, a Tanh activation, and a
    per‑channel affine shift that emulates displacement.
    """
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

        def forward(self, inp: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            out = self.activation(self.linear(inp))
            out = out * self.scale + self.shift
            return out

    return Layer()


class FraudDetectionHybrid(nn.Module):
    """
    Classic hybrid model: a stack of photonic‑style layers followed by a
    lightweight neural head.  The head can be tuned to any hidden
    dimension and activation, giving the user flexibility while still
    keeping the core physics‑inspired feature extractor unchanged.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        # build the physics‑inspired backbone
        backbone = [_layer_from_params(input_params, clip=False)]
        backbone.extend(_layer_from_params(l, clip=True) for l in layers)
        backbone.append(nn.Linear(2, 1))  # final linear for regression

        self.backbone = nn.Sequential(*backbone)

        # default head: single linear layer with no activation
        hidden_dims = hidden_dims or [32, 16]
        head_layers = []
        in_dim = 1
        for h in hidden_dims:
            head_layers.append(nn.Linear(in_dim, h))
            head_layers.append(nn.Tanh())
            in_dim = h
        head_layers.append(nn.Linear(in_dim, 1))
        self.head = nn.Sequential(*head_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return self.head(x)


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """
    Exposed helper that reproduces the original sequential
    construction for compatibility with legacy code.
    """
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(l, clip=True) for l in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


__all__ = [
    "FraudLayerParameters",
    "FraudDetectionHybrid",
    "build_fraud_detection_program",
]
