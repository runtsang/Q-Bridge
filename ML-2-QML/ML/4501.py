"""Hybrid classical estimator that unifies autoencoder, fraud‑detection, and sampler concepts."""

from __future__ import annotations

import torch
from torch import nn
from dataclasses import dataclass
from typing import Tuple

@dataclass
class FraudLayerParameters:
    """Parameters for a fraud‑detection inspired linear layer."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

class FraudLayer(nn.Module):
    """Custom layer combining a linear transform, activation, and affine scaling."""
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
        self.register_buffer("scale", torch.tensor(params.displacement_r, dtype=torch.float32))
        self.register_buffer("shift", torch.tensor(params.displacement_phi, dtype=torch.float32))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(self.linear(x))
        return out * self.scale + self.shift

class AutoencoderEncoder(nn.Module):
    """Encoder part of a lightweight autoencoder."""
    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: Tuple[int, int] = (128, 64)) -> None:
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*layers)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

class HybridEstimatorQNN(nn.Module):
    """Hybrid estimator that stacks an autoencoder encoder, a fraud layer, and a linear head."""
    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 32,
                 hidden_dims: Tuple[int, int] = (128, 64),
                 fraud_params: FraudLayerParameters | None = None) -> None:
        super().__init__()
        self.encoder = AutoencoderEncoder(input_dim, latent_dim, hidden_dims)
        default_params = FraudLayerParameters(
            bs_theta=0.0, bs_phi=0.0,
            phases=(0.0, 0.0),
            squeeze_r=(0.0, 0.0),
            squeeze_phi=(0.0, 0.0),
            displacement_r=(0.0, 0.0),
            displacement_phi=(0.0, 0.0),
            kerr=(0.0, 0.0)
        )
        self.fraud_layer = FraudLayer(fraud_params or default_params)
        self.head = nn.Linear(2, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        z2 = z[:, :2]  # feed first two latent dims into fraud layer
        fraud_out = self.fraud_layer(z2)
        return self.head(fraud_out)

__all__ = ["HybridEstimatorQNN", "FraudLayerParameters", "FraudLayer", "AutoencoderEncoder"]
