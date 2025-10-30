"""
Combined sampler and fraud‑detection model for classical training.
"""

from __future__ import annotations

import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable, Sequence

# ------------------------------------------------------------------
# Fraud‑detection layer utilities (adapted from the photonic seed)
# ------------------------------------------------------------------
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
            return outputs * self.scale + self.shift

    return Layer()

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """
    Create a sequential PyTorch model mirroring the layered structure.
    The first layer is un‑clipped; subsequent layers are clipped to keep
    parameters within a stable range.
    """
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# ------------------------------------------------------------------
# Combined SamplerQNN architecture
# ------------------------------------------------------------------
class SamplerQNN(nn.Module):
    """
    A hybrid network that first performs a simple softmax sampler
    and then feeds the result through a fraud‑detection pipeline.
    """
    def __init__(
        self,
        fraud_input: FraudLayerParameters,
        fraud_layers: Sequence[FraudLayerParameters],
    ) -> None:
        super().__init__()
        # Sampler core
        self.sampler = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )
        # Fraud‑detection stack
        self.fraud_net = build_fraud_detection_program(fraud_input, fraud_layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: 2‑dim input → sampler → fraud detector → scalar output.
        """
        x = torch.softmax(self.sampler(inputs), dim=-1)
        return self.fraud_net(x)

__all__ = ["FraudLayerParameters", "build_fraud_detection_program", "SamplerQNN"]
