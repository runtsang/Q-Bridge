"""Hybrid fraud detection model with a trainable classical encoder and a quantum photonic circuit."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn


# --------------------------------------------------------------------------- #
# Classical encoder
# --------------------------------------------------------------------------- #
@dataclass
class FraudEncoderParameters:
    """Parameters describing a small fully‑connected encoder that maps input features to a latent space."""
    hidden_dim: int
    out_dim: int
    bias: bool = False


def _build_encoder(
    input_dim: int,
    encoder_params: FraudEncoderParameters,
) -> nn.Module:
    """Construct a small MLP encoder that is used before the quantum circuit."""
    layers: Sequence[nn.Module] = [
        nn.Linear(input_dim, encoder_params.hidden_dim),
        nn.ReLU(),
        nn.Linear(encoder_params.hidden_dim, encoder_params.out_dim),
    ]
    if not encoder_params.bias:
        for m in layers:
            if isinstance(m, nn.Linear):
                m.bias.requires_grad = False
    return nn.Sequential(*layers)


# --------------------------------------------------------------------------- #
# Classical fraud layer
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParameters:
    """Parameters describing a fully connected layer in the classical part of the model."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


def _clip(value: float, bound: float) -> float:
    """Clip values to keep the photonic parameters within a safe range."""
    return max(-bound, min(bound, value))


def _layer_from_params(
    params: FraudLayerParameters,
    *,
    clip: bool,
) -> nn.Module:
    """Create a one‑layer neural network that mirrors the photonic layer."""
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


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    encoder_params: FraudEncoderParameters,
    input_dim: int = 2,
) -> nn.Sequential:
    """Create a sequential PyTorch model that first encodes the input, then applies the photonic‑style layers, and finally outputs a binary score."""
    encoder = _build_encoder(input_dim, encoder_params)
    modules = [encoder]
    modules.append(_layer_from_params(input_params, clip=False))
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


__all__ = ["FraudEncoderParameters", "FraudLayerParameters", "build_fraud_detection_program"]
