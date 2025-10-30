"""HybridSamplerQNN: classical model inspired by SamplerQNN and FraudDetection.

The network starts with a 2→4 linear layer followed by a configurable number of
Fraud‑style layers. Each Fraud layer contains a linear transform, a tanh
activation, and a learnable scale/shift that mimics the displacement and
squeezing terms from the photonic implementation. The final layer maps to a
single output which is soft‑maxed to produce a probability distribution over
two classes.

This design allows the model to capture both simple linear relationships
(through the initial layer) and more expressive nonlinear transformations
(through the Fraud layers), while keeping the parameter count modest.
"""

from __future__ import annotations

import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable, List, Tuple


@dataclass
class FraudLayerParameters:
    """Parameters for a single Fraud‑style layer."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]


def _fraud_layer(params: FraudLayerParameters, clip: bool = True) -> nn.Module:
    """Create a Fraud‑style linear block with optional clipping."""
    weight = torch.tensor(
        [[params.bs_theta, params.bs_phi],
         [params.squeeze_r[0], params.squeeze_r[1]]],
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

    scale = torch.tensor(params.displacement_r, dtype=torch.float32)
    shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = nn.Tanh()
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            y = self.activation(self.linear(x))
            return y * self.scale + self.shift

    return Layer()


class HybridSamplerQNN(nn.Module):
    """A hybrid classical sampler network combining simple and fraud‑style layers."""

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layer_params: Iterable[FraudLayerParameters],
    ) -> None:
        super().__init__()
        modules: List[nn.Module] = [nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )]
        modules.append(_fraud_layer(input_params, clip=False))
        modules.extend(_fraud_layer(p, clip=True) for p in layer_params)
        modules.append(nn.Linear(2, 1))
        self.model = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        logits = self.model(x)
        return torch.softmax(logits, dim=-1)
