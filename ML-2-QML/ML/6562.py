"""Hybrid classical sampler network with fraud‑detection inspired preprocessing."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Iterable

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

def _layer_from_params(params: FraudLayerParameters, *, clip: bool = False) -> nn.Module:
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

class SamplerQNN(nn.Module):
    """Hybrid classical sampler network with fraud‑detection inspired preprocessing."""
    def __init__(self, fraud_params: FraudLayerParameters, fraud_clip: bool = True):
        super().__init__()
        # Fraud layer
        self.fraud_layer = _layer_from_params(fraud_params, clip=fraud_clip)
        # Sampler network: 2→4→2
        self.sampler_net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Pass through fraud layer then sampler network, output probability distribution
        x = self.fraud_layer(inputs)
        x = self.sampler_net(x)
        return F.softmax(x, dim=-1)

    def get_quantum_weights(self) -> dict[str, torch.Tensor]:
        """Return a dict of weight tensors that can be mapped to quantum parameters."""
        return {
            "fraud_weights": self.fraud_layer.linear.weight.detach(),
            "fraud_bias": self.fraud_layer.linear.bias.detach(),
            "sampler_weights_1": self.sampler_net[0].weight.detach(),
            "sampler_bias_1": self.sampler_net[0].bias.detach(),
            "sampler_weights_2": self.sampler_net[2].weight.detach(),
            "sampler_bias_2": self.sampler_net[2].bias.detach(),
        }

__all__ = ["SamplerQNN", "FraudLayerParameters", "_layer_from_params"]
