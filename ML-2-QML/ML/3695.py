"""Hybrid classical attention and fraud detection model.

This module merges the self‑attention mechanism from the original
SelfAttention seed with the fraud‑detection architecture from the
FraudDetection seed.  The resulting nn.Module can be trained end‑to‑end
and serves as a drop‑in replacement for either component, while
providing a unified interface for research experiments.
"""

import torch
from torch import nn
import numpy as np
from dataclasses import dataclass
from typing import Iterable

# --- Self‑attention --------------------------------------------
class ClassicalSelfAttention(nn.Module):
    """Pure‑Python/Torch self‑attention block used as a drop‑in
    replacement for the original SelfAttention implementation.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        # Linear projections for query/key/value
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, rotation_params: torch.Tensor,
                entangle_params: torch.Tensor) -> torch.Tensor:
        # Projection matrices are parameterised by the supplied arrays
        # to keep the interface identical to the seed.
        w_q = rotation_params.reshape(self.embed_dim, self.embed_dim)
        w_k = entangle_params.reshape(self.embed_dim, self.embed_dim)
        q = torch.matmul(x, w_q)
        k = torch.matmul(x, w_k)
        v = x  # value is the input itself
        scores = torch.softmax(torch.matmul(q, k.T) / np.sqrt(self.embed_dim),
                               dim=-1)
        return torch.matmul(scores, v)

# --- Fraud‑detection --------------------------------------------
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

# --- Hybrid model -----------------------------------------------
class HybridAttentionFraudDetector(nn.Module):
    """Hybrid classical model that first applies self‑attention and then
    performs fraud detection on the attended representations.
    """
    def __init__(
        self,
        attention_dim: int,
        fraud_input: FraudLayerParameters,
        fraud_layers: Iterable[FraudLayerParameters],
    ):
        super().__init__()
        self.attention = ClassicalSelfAttention(attention_dim)
        self.fraud = build_fraud_detection_program(fraud_input, fraud_layers)

    def forward(
        self,
        inputs: torch.Tensor,
        rotation_params: torch.Tensor,
        entangle_params: torch.Tensor,
    ) -> torch.Tensor:
        attn_out = self.attention(inputs, rotation_params, entangle_params)
        return self.fraud(attn_out)

__all__ = [
    "HybridAttentionFraudDetector",
    "FraudLayerParameters",
    "build_fraud_detection_program",
]
