"""Hybrid SamplerQNN__gen294 with classical encoder, fraud layer, and kernel.

This module builds upon the original SamplerQNN by
- adding an auto‑encoder encoder that maps raw inputs to a compact latent space;
- passing the latent representation through a fraud‑detection style
  fully‑connected layer that mimics photonic gates;
- exposing a radial‑basis‑function kernel for similarity analysis.
The resulting feature vector can be fed into the quantum sampler defined in
:mod:`qml_code` to obtain a probability distribution.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple

# ----------------------------------------------------
# Fraud detection layer (classical analogue)
# ----------------------------------------------------
@dataclass
class FraudLayerParameters:
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

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()

class FraudLayer(nn.Module):
    """Wraps a single fraud‑detection style layer."""
    def __init__(self, params: FraudLayerParameters) -> None:
        super().__init__()
        self.module = _layer_from_params(params, clip=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.module(x)

# ----------------------------------------------------
# Auto‑encoder encoder
# ----------------------------------------------------
@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class AutoencoderNet(nn.Module):
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        encoder_layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))

# ----------------------------------------------------
# RBF kernel
# ----------------------------------------------------
class Kernel(nn.Module):
    """Radial‑basis‑function kernel for similarity."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * (diff * diff).sum(-1, keepdim=True))

# ----------------------------------------------------
# Hybrid SamplerQNN__gen294
# ----------------------------------------------------
class SamplerQNN__gen294(nn.Module):
    """Hybrid classical‑quantum sampler that prepares features for the
    quantum sampler defined in the QML module.
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: Tuple[int, int] = (128, 64),
        dropout: float = 0.1,
        fraud_params: FraudLayerParameters | None = None,
        kernel_gamma: float = 1.0,
    ) -> None:
        super().__init__()
        self.encoder = AutoencoderNet(
            AutoencoderConfig(
                input_dim=input_dim,
                latent_dim=latent_dim,
                hidden_dims=hidden_dims,
                dropout=dropout,
            )
        )
        self.fraud = FraudLayer(fraud_params) if fraud_params else nn.Identity()
        self.kernel = Kernel(gamma=kernel_gamma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Return fraud‑layer transformed latent features."""
        latent = self.encoder.encode(x)
        return self.fraud(latent)

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute Gram matrix using the RBF kernel."""
        return self.kernel(a, b)

__all__ = [
    "FraudLayerParameters",
    "FraudLayer",
    "AutoencoderConfig",
    "AutoencoderNet",
    "Kernel",
    "SamplerQNN__gen294",
]
