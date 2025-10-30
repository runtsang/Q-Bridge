from __future__ import annotations

import numpy as np
import torch
from torch import nn
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

class SelfAttentionLayer(nn.Module):
    """Learnable self‑attention block that operates on a 2‑dimensional fraud feature vector."""
    def __init__(self, embed_dim: int = 4) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.rotation = nn.Parameter(torch.randn(embed_dim * 3))
        self.entangle = nn.Parameter(torch.randn(embed_dim - 1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        query = torch.matmul(inputs, self.rotation.view(self.embed_dim, -1))
        key = torch.matmul(inputs, self.entangle.view(self.embed_dim, -1))
        value = inputs
        scores = torch.softmax(query @ key.t() / np.sqrt(self.embed_dim), dim=-1)
        return torch.matmul(scores, value)

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    attention: bool = True,
) -> nn.Sequential:
    """Construct a hybrid classical fraud‑detection model with optional self‑attention."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    if attention:
        modules.append(SelfAttentionLayer())
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

__all__ = ["FraudLayerParameters", "build_fraud_detection_program", "SelfAttentionLayer"]
