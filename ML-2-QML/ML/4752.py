"""Hybrid binary classifier implemented entirely in PyTorch, integrating a classical convolution filter and a fraud‑detection style feed‑forward network."""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

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

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
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

class ConvFilter(nn.Module):
    """Simple 2‑D convolution that emulates the quanvolution filter."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        # Output two channels to match the fraud‑detection input
        self.conv = nn.Conv2d(1, 2, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        # Reduce spatial dimensions by averaging
        return activations.mean(dim=[2, 3])

class HybridBinaryClassifier(nn.Module):
    """End‑to‑end classical binary classifier that fuses a convolution filter and a fraud‑detection style network."""
    def __init__(
        self,
        conv_kernel: int = 2,
        conv_threshold: float = 0.0,
        fraud_params: Optional[Iterable[FraudLayerParameters]] = None,
    ) -> None:
        super().__init__()
        self.conv = ConvFilter(kernel_size=conv_kernel, threshold=conv_threshold)
        if fraud_params is not None:
            layers = list(fraud_params)
            self.fraud = build_fraud_detection_program(layers[0], layers[1:])
        else:
            self.fraud = nn.Identity()
        self.final = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect input shape (batch, 1, H, W)
        x = self.conv(x)
        x = self.fraud(x)
        logits = self.final(x)
        probs = torch.sigmoid(logits)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "ConvFilter",
    "HybridBinaryClassifier",
]
