from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Iterable, Sequence

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
        dtype=torch.float32
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

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    output_dim: int = 10,
) -> nn.Sequential:
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, output_dim))
    return nn.Sequential(*modules)

class HybridQuanvolutionFilter(nn.Module):
    """Classical convolutional filter that mimics the 2×2 patch extraction of the quantum filter."""
    def __init__(self, in_channels: int = 1, out_channels: int = 4, kernel_size: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.conv(x)
        return features.view(x.size(0), -1)

class AttentionBlock(nn.Module):
    """Learnable self‑attention over a 2‑dimensional feature vector."""
    def __init__(self, embed_dim: int = 2) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key   = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # x: (batch, seq_len=1, embed_dim)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scores = torch.softmax(q @ k.transpose(-2, -1) / np.sqrt(self.embed_dim), dim=-1)
        return scores @ v

class HybridQuanvolutionClassifier(nn.Module):
    """Hybrid classical‑quantum network that combines a quanvolution filter, a self‑attention block,
    and a fraud‑detection style linear stack before the final classification head."""
    def __init__(self, fraud_params: Iterable[FraudLayerParameters]) -> None:
        super().__init__()
        self.filter = HybridQuanvolutionFilter()
        self.proj   = nn.Linear(4 * 14 * 14, 2)
        self.attn   = AttentionBlock(embed_dim=2)
        self.fraud  = build_fraud_detection_program(fraud_params[0], fraud_params[1:], output_dim=10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        feat = self.filter(x)
        proj = self.proj(feat)
        attn_out = self.attn(proj.unsqueeze(1)).squeeze(1)
        logits = self.fraud(attn_out)
        return F.log_softmax(logits, dim=-1)

__all__ = ["HybridQuanvolutionClassifier", "FraudLayerParameters", "build_fraud_detection_program"]
