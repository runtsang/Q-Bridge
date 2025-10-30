"""Hybrid fraud detection model combining classical photonic-inspired layers
and a quantum sampler head.

The model mirrors the photonic structure defined in the original
FraudDetection seed while adding a sampler network that outputs
class probabilities.  The architecture is fully trainable in PyTorch
and can be paired with a quantum sampler during inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class FraudLayerParams:
    """Parameters for a single photonic-inspired dense layer."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def _layer_from_params(params: FraudLayerParams, clip: bool = False) -> nn.Module:
    """Build a dense layer that emulates a photonic layer."""
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
            out = self.activation(self.linear(inputs))
            out = out * self.scale + self.shift
            return out

    return Layer()


class SamplerModule(nn.Module):
    """Simple softmax classifier used as a sampler head."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return F.softmax(self.net(inputs), dim=-1)


class FraudDetectionHybrid(nn.Module):
    """Hybrid fraud detection model with photonic-inspired layers and a sampler head."""
    def __init__(
        self,
        input_params: FraudLayerParams,
        hidden_params: Iterable[FraudLayerParams],
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [_layer_from_params(input_params, clip=False)]
            + [_layer_from_params(p, clip=True) for p in hidden_params]
        )
        self.output_linear = nn.Linear(2, 1)
        self.sampler = SamplerModule()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            x = layer(x)
        logits = self.output_linear(x)
        probs = self.sampler(x)
        return logits, probs


__all__ = ["FraudLayerParams", "FraudDetectionHybrid"]
