"""Hybrid classical estimator combining self‑attention, sampler, and fraud‑detection layers."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Iterable, Sequence, List


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


class ClassicalSelfAttention(nn.Module):
    """Simple self‑attention block mimicking the quantum interface."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)
        scores = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.embed_dim), dim=-1)
        return torch.matmul(scores, v)


class SamplerModule(nn.Module):
    """Probabilistic sampler mirroring the quantum SamplerQNN."""

    def __init__(self, in_features: int = 2, out_features: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 4),
            nn.Tanh(),
            nn.Linear(4, out_features)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(inputs), dim=-1)


class FraudLayer(nn.Module):
    """A single fraud‑detection layer with trainable parameters."""

    def __init__(self, params: FraudLayerParameters, clip: bool = True) -> None:
        super().__init__()
        weight = torch.tensor([[params.bs_theta, params.bs_phi],
                               [params.squeeze_r[0], params.squeeze_r[1]]], dtype=torch.float32)
        bias = torch.tensor(params.phases, dtype=torch.float32)
        if clip:
            weight = weight.clamp(-5.0, 5.0)
            bias = bias.clamp(-5.0, 5.0)
        self.linear = nn.Linear(2, 2)
        with torch.no_grad():
            self.linear.weight.copy_(weight)
            self.linear.bias.copy_(bias)
        self.activation = nn.Tanh()
        self.scale = torch.tensor(params.displacement_r, dtype=torch.float32)
        self.shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out = self.activation(self.linear(inputs))
        return out * self.scale + self.shift


class FraudDetectionModel(nn.Module):
    """Sequential model built from multiple fraud layers."""

    def __init__(self, input_params: FraudLayerParameters, layers: Iterable[FraudLayerParameters]) -> None:
        super().__init__()
        modules: List[nn.Module] = [FraudLayer(input_params, clip=False)]
        modules.extend(FraudLayer(l, clip=True) for l in layers)
        modules.append(nn.Linear(2, 1))
        self.model = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class HybridEstimator(nn.Module):
    """Comprehensive classical estimator combining attention, sampler, and fraud detection."""

    def __init__(self, attention_dim: int, fraud_layers: Sequence[FraudLayerParameters]) -> None:
        super().__init__()
        self.attention = ClassicalSelfAttention(attention_dim)
        self.sampler = SamplerModule()
        self.fraud = FraudDetectionModel(fraud_layers[0], fraud_layers[1:])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attention(x)
        x = self.sampler(x)
        return self.fraud(x)

    def evaluate(self, inputs: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper that runs the model in eval mode."""
        self.eval()
        with torch.no_grad():
            return self.forward(inputs)

__all__ = ["HybridEstimator", "FraudLayerParameters"]
