"""
ConvFraudHybrid: Classical module that fuses a convolutional filter with a fraud detection network.
"""

from __future__ import annotations

import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable, Sequence

# ----------------------------------------------------------------------
# Convolutional filter
# ----------------------------------------------------------------------
class ConvFilter(nn.Module):
    """
    A lightweight 2‑D convolutional filter that emulates a quantum filter.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=(2, 3))

# ----------------------------------------------------------------------
# Fraud detection block
# ----------------------------------------------------------------------
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
    return max(-bound, min(value, bound))

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
            out = self.activation(self.linear(inputs))
            out = out * self.scale + self.shift
            return out

    return Layer()

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# ----------------------------------------------------------------------
# Combined hybrid model
# ----------------------------------------------------------------------
class ConvFraudHybrid(nn.Module):
    """
    Combines a 2‑D convolutional filter with a fraud‑detection neural network.
    """

    def __init__(
        self,
        conv_kernel_size: int = 2,
        conv_threshold: float = 0.0,
        fraud_input_params: FraudLayerParameters | None = None,
        fraud_layers: Iterable[FraudLayerParameters] | None = None,
    ) -> None:
        super().__init__()
        self.conv = ConvFilter(kernel_size=conv_kernel_size, threshold=conv_threshold)
        self.fraud_net = build_fraud_detection_program(
            fraud_input_params if fraud_input_params else FraudLayerParameters(
                bs_theta=0.0,
                bs_phi=0.0,
                phases=(0.0, 0.0),
                squeeze_r=(0.0, 0.0),
                squeeze_phi=(0.0, 0.0),
                displacement_r=(0.0, 0.0),
                displacement_phi=(0.0, 0.0),
                kerr=(0.0, 0.0),
            ),
            fraud_layers if fraud_layers is not None else [],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that applies the convolution filter and then the fraud detection network.
        """
        conv_out = self.conv(x)  # shape (batch, 1)
        img_mean = x.mean(dim=(2, 3), keepdim=True)  # shape (batch, 1)
        features = torch.cat([conv_out, img_mean], dim=1)  # shape (batch, 2)
        return self.fraud_net(features)

__all__ = ["ConvFraudHybrid", "FraudLayerParameters", "build_fraud_detection_program"]
