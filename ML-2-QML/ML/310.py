"""Enhanced classical fraud detection model with early‑exit and attention mechanisms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn


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
    # New: weight for the attention‑style gate
    attention_weight: float = 1.0


def _clip(value: float, bound: float) -> float:
    """Clamp value to the interval [-bound, bound]."""
    return max(-bound, min(bound, value))


def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    """Create a single layer from a FraudLayerParameters instance."""
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
            self.attention_weight = torch.tensor(params.attention_weight)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.attention_weight
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Create a sequential PyTorch model mirroring the layered structure."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


class FraudDetection__gen343(nn.Module):
    """Classical fraud detection model with early‑exit and attention gating."""
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        early_exit_threshold: float = 0.9,
        attention_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.model = build_fraud_detection_program(input_params, layers)
        self.threshold = early_exit_threshold
        self.attention_weight = attention_weight
        # Override attention weight in all layers
        for layer in self.model:
            if hasattr(layer, "attention_weight"):
                layer.attention_weight = torch.tensor(attention_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.model:
            out = layer(out)
            # Early‑exit check: confidence from sigmoid
            confidence = torch.sigmoid(out)
            if torch.all(confidence > self.threshold):
                break
        return out

    def set_threshold(self, threshold: float) -> None:
        self.threshold = threshold

    def set_attention_weight(self, weight: float) -> None:
        self.attention_weight = weight
        for layer in self.model:
            if hasattr(layer, "attention_weight"):
                layer.attention_weight = torch.tensor(weight)


__all__ = ["FraudLayerParameters", "build_fraud_detection_program", "FraudDetection__gen343"]
