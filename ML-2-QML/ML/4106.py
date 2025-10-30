from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Iterable


@dataclass
class FraudLayerParameters:
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

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


class ClassicalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int = 4) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.query_linear = nn.Linear(embed_dim, embed_dim)
        self.key_linear = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> torch.Tensor:
        # Convert numpy parameters to tensors
        rot = torch.tensor(rotation_params, dtype=torch.float32)
        ent = torch.tensor(entangle_params, dtype=torch.float32)

        query = self.query_linear(inputs @ rot.reshape(self.embed_dim, -1))
        key = self.key_linear(inputs @ ent.reshape(self.embed_dim, -1))
        scores = F.softmax(query @ key.transpose(-1, -2) / np.sqrt(self.embed_dim), dim=-1)
        return scores @ inputs


class HybridAttentionFraudNet(nn.Module):
    def __init__(
        self,
        attention: nn.Module,
        fraud_params: FraudLayerParameters,
        fraud_layers: list[FraudLayerParameters],
    ) -> None:
        super().__init__()
        self.attention = attention
        self.fraud_net = build_fraud_detection_program(fraud_params, fraud_layers)

    def forward(
        self,
        inputs: torch.Tensor,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> torch.Tensor:
        attn_out = self.attention(inputs, rotation_params, entangle_params)
        return self.fraud_net(attn_out)


__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "ClassicalSelfAttention",
    "HybridAttentionFraudNet",
]
