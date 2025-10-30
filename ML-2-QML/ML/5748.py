"""Hybrid quanvolution + fraud detection classifier.

This module extends the original classical quanvolution example by
adding a fraud‑detection style linear head.  The head is built from
parameterised 2‑input layers that mimic the photonic circuit used in
the QML seed.  The combined architecture is useful for experiments
where a quantum‑inspired feature extractor is paired with a
classical fraud‑detection pipeline.

Classes
-------
HybridQuanvolutionFraudClassifier
    A two‑stage model: classical quanvolution filter followed by a
    fraud‑detection style sequential head.
QuanvolutionFilter, QuanvolutionClassifier
    Backwards‑compatible wrappers around the original architecture.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Iterable, Tuple

# Classical quanvolution filter --------------------------------------------

class QuanvolutionFilter(nn.Module):
    """Simple 2×2 stride‑2 convolution that flattens the output."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.conv(x)
        return features.view(x.size(0), -1)

class QuanvolutionClassifier(nn.Module):
    """Legacy classifier that uses the classical filter and a linear head."""
    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

# Fraud‑detection style head ----------------------------------------------

@dataclass
class FraudLayerParameters:
    """Parameters for a single fraud‑detection style linear layer."""
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

def build_fraud_detection_head(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Create a sequential PyTorch model mirroring the fraud‑detection layers."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# Hybrid model --------------------------------------------------------------

class HybridQuanvolutionFraudClassifier(nn.Module):
    """
    A hybrid model that combines a classical quanvolution filter with
    a fraud‑detection style linear head.

    Parameters
    ----------
    fraud_input_params : FraudLayerParameters, optional
        Parameters for the first fraud layer.  When *None* a default
        parameter set is used.
    fraud_layer_params : Iterable[FraudLayerParameters], optional
        Parameters for subsequent fraud layers.  Defaults to a short
        sequence that demonstrates the construction of the head.
    """
    def __init__(
        self,
        fraud_input_params: FraudLayerParameters | None = None,
        fraud_layer_params: Iterable[FraudLayerParameters] | None = None,
    ) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()

        if fraud_input_params is None:
            fraud_input_params = FraudLayerParameters(
                bs_theta=0.0,
                bs_phi=0.0,
                phases=(0.0, 0.0),
                squeeze_r=(0.0, 0.0),
                squeeze_phi=(0.0, 0.0),
                displacement_r=(1.0, 1.0),
                displacement_phi=(0.0, 0.0),
                kerr=(0.0, 0.0),
            )
        if fraud_layer_params is None:
            fraud_layer_params = [
                FraudLayerParameters(
                    bs_theta=0.1,
                    bs_phi=0.2,
                    phases=(0.1, 0.1),
                    squeeze_r=(0.1, 0.1),
                    squeeze_phi=(0.0, 0.0),
                    displacement_r=(0.5, 0.5),
                    displacement_phi=(0.0, 0.0),
                    kerr=(0.0, 0.0),
                )
            ]

        self.fraud_head = build_fraud_detection_head(
            fraud_input_params, fraud_layer_params
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.qfilter(x)
        logits = self.fraud_head(features)
        return F.log_softmax(logits, dim=-1)

# Backwards‑compatible aliases ---------------------------------------------

QuanvolutionFilterClass = QuanvolutionFilter
QuanvolutionClassifierClass = QuanvolutionClassifier

__all__ = [
    "HybridQuanvolutionFraudClassifier",
    "QuanvolutionFilter",
    "QuanvolutionClassifier",
    "FraudLayerParameters",
    "build_fraud_detection_head",
]
