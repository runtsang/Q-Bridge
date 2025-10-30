"""PyTorch implementation of a deep residual fraud‑detection network.

The model mirrors the photonic architecture but adds batch‑normalisation,
dropout, and residual connections to improve generalisation.  The
`FraudDetectionHybrid` class is fully self‑contained and can be used
directly with standard PyTorch training loops.

Key features
------------
* Parameterised 2‑D linear layers with optional clipping.
* Batch‑norm and dropout after each linear block.
* Residual (skip) connections across the main body of the network.
* Single output neuron with sigmoid activation for binary fraud scoring.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn
from torch.nn.functional import sigmoid

# --------------------------------------------------------------------------- #
#  Parameter dataclass
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParameters:
    """Parameters that describe a single 2‑D linear block."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]
    dropout: float = 0.0  # new hyper‑parameter

# --------------------------------------------------------------------------- #
#  Helper functions
# --------------------------------------------------------------------------- #
def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _layer_from_params(
    params: FraudLayerParameters, *, clip: bool, dropout: float
) -> nn.Module:
    # Build the weight matrix from the photonic parameters
    weight = torch.tensor(
        [[params.bs_theta, params.bs_phi],
         [params.squeeze_r[0], params.squeeze_r[1]]],
        dtype=torch.float32,
    )
    bias = torch.tensor(params.phases, dtype=torch.float32)

    if clip:
        weight = weight.clamp(-5.0, 5.0)
        bias = bias.clamp(-5.0, 5.0)

    linear = nn.Linear(2, 2, bias=True)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)

    # Activation and scaling
    activation = nn.Tanh()
    scale = torch.tensor(params.displacement_r, dtype=torch.float32)
    shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.norm = nn.BatchNorm1d(2)
            self.dropout = nn.Dropout(dropout)
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = self.linear(x)
            y = self.activation(y)
            y = self.norm(y)
            y = self.dropout(y)
            y = y * self.scale + self.shift
            return y

    return Layer()

# --------------------------------------------------------------------------- #
#  Model definition
# --------------------------------------------------------------------------- #
class FraudDetectionHybrid(nn.Module):
    """Deep residual network for fraud detection."""
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
    ) -> None:
        super().__init__()
        # First layer is un‑clipped
        self.input_layer = _layer_from_params(input_params, clip=False, dropout=input_params.dropout)

        # Residual blocks
        residuals: list[nn.Module] = []
        for layer in layers:
            residuals.append(
                _layer_from_params(layer, clip=True, dropout=layer.dropout)
            )
        self.residuals = nn.ModuleList(residuals)

        # Final classifier
        self.classifier = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.input_layer(x)
        for block in self.residuals:
            # skip connection
            out = out + block(out)
        logits = self.classifier(out)
        return sigmoid(logits)

# --------------------------------------------------------------------------- #
#  Factory function
# --------------------------------------------------------------------------- #
def build_fraud_detection_model(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> FraudDetectionHybrid:
    """Convenience constructor mirroring the original API."""
    return FraudDetectionHybrid(input_params, layers)

__all__ = ["FraudLayerParameters", "FraudDetectionHybrid", "build_fraud_detection_model"]
