from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Iterable, Tuple

# --------------------------------------------------------------------------- #
# Fraud‑detection style layer – classical implementation
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Classical quanvolution filter
# --------------------------------------------------------------------------- #
class QuanvolutionFilter(nn.Module):
    """Convolutional filter that extracts 2×2 patches and flattens them."""
    def __init__(self, in_channels: int = 1, out_channels: int = 4, kernel_size: int = 2, stride: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.conv(x).view(x.size(0), -1)


# --------------------------------------------------------------------------- #
# Classical hybrid head
# --------------------------------------------------------------------------- #
class HybridHead(nn.Module):
    """Simple dense head that optionally applies a sigmoid shift."""
    def __init__(self, in_features: int, out_features: int = 10, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        logits = self.linear(x)
        return torch.sigmoid(logits + self.shift)


# --------------------------------------------------------------------------- #
# Combined classical network
# --------------------------------------------------------------------------- #
class QuanvolutionClassifier(nn.Module):
    """Full pipeline: quanvolution filter → fraud‑detection layers → hybrid head."""
    def __init__(
        self,
        num_classes: int = 10,
        fraud_layers: Iterable[FraudLayerParameters] | None = None,
        shift: float = 0.0,
    ) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.fraud_layers = nn.ModuleList(
            [_layer_from_params(p, clip=True) for p in (fraud_layers or [])]
        )
        # Compute feature size after filter and fraud layers
        dummy = torch.zeros(1, 1, 28, 28)
        feat = self.qfilter(dummy)
        for layer in self.fraud_layers:
            feat = layer(feat)
        in_features = feat.size(1)
        self.head = HybridHead(in_features, out_features=num_classes, shift=shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.qfilter(x)
        for layer in self.fraud_layers:
            features = layer(features)
        return self.head(features)


__all__ = [
    "FraudLayerParameters",
    "_layer_from_params",
    "QuanvolutionFilter",
    "HybridHead",
    "QuanvolutionClassifier",
]
