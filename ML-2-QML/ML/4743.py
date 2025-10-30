"""
Classical implementation of a hybrid quanvolution network.
It combines a 2×2 convolutional filter with a fraud‑detection style
feed‑forward head that mimics the photonic circuit of the QML seed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Iterable, Sequence

__all__ = ["FraudLayerParameters", "build_fraud_detection_program", "QuanvolutionFilterClassic", "QuanvolutionHybridModel"]


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
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Create a sequential PyTorch model mirroring the layered structure."""
    modules: list[nn.Module] = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


class QuanvolutionFilterClassic(nn.Module):
    """
    Classical 2×2 convolutional filter that downsamples an 28×28 image to 14×14 patches.
    """
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)  # flatten to (batch, 4*14*14)


class QuanvolutionHybridModel(nn.Module):
    """
    Hybrid classifier that chains the classical quanvolution filter with a
    fraud‑detection style head.  The head can be configured with a variable
    number of layers to mimic the depth of the quantum ansatz from the QML seed.
    """
    def __init__(self, depth: int = 3, num_classes: int = 10) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilterClassic()
        # Build a simple fraud‑detection head with random parameters.
        base_params = FraudLayerParameters(
            bs_theta=0.5, bs_phi=0.3,
            phases=(0.1, -0.1),
            squeeze_r=(0.2, 0.2), squeeze_phi=(0.0, 0.0),
            displacement_r=(0.5, 0.5), displacement_phi=(0.0, 0.0),
            kerr=(0.0, 0.0)
        )
        layer_params = [
            FraudLayerParameters(
                bs_theta=0.4, bs_phi=0.2,
                phases=(0.05, -0.05),
                squeeze_r=(0.1, 0.1), squeeze_phi=(0.0, 0.0),
                displacement_r=(0.4, 0.4), displacement_phi=(0.0, 0.0),
                kerr=(0.0, 0.0)
            )
            for _ in range(depth)
        ]
        self.classifier_head = build_fraud_detection_program(base_params, layer_params)
        self.final_linear = nn.Linear(2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        # The fraud‑detection head expects 2‑dimensional inputs; collapse spatial dims.
        reduced = features.view(features.size(0), 2)
        logits = self.classifier_head(reduced)
        logits = self.final_linear(logits)
        return torch.log_softmax(logits, dim=-1)
