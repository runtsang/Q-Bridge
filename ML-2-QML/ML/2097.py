"""
Classical fraud detection model with residual connections and modern regularisation.

The model is a direct extension of the seed implementation:
* Each layer now includes batch‑normalisation, dropout and a residual add‑on.
* A final linear head produces a single logit for binary classification.
* The public API mirrors the original: `FraudLayerParameters`, a builder
  `build_fraud_detection_model` and a convenience wrapper `FraudDetectionHybrid`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import nn


@dataclass
class FraudLayerParameters:
    """Parameters describing a fully connected layer in the classical model."""
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

    linear = nn.Linear(2, 2, bias=True)
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
            self.dropout = nn.Dropout(p=0.2)
            self.batch_norm = nn.BatchNorm1d(2)
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            out = self.linear(inputs)
            out = self.activation(out)
            out = self.dropout(out)
            out = self.batch_norm(out)
            out = out * self.scale + self.shift
            return out

    return Layer()


def build_fraud_detection_model(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Module:
    """Return a PyTorch model that matches the photonic architecture
    but includes residual connections between each layer and the input.
    """
    modules = [_layer_from_params(input_params, clip=False)]

    for layer_params in layers:
        modules.append(_layer_from_params(layer_params, clip=True))

    modules.append(nn.Linear(2, 1))

    class ResidualNet(nn.Module):
        def __init__(self, body: nn.ModuleList):
            super().__init__()
            self.body = body

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = x
            for layer in self.body:
                out = layer(out) + out  # residual connection
            out = self.body[-1](out)
            return out

    return ResidualNet(nn.ModuleList(modules))


class FraudDetectionHybrid:
    """Convenience wrapper around the model and its loss."""
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.criterion = nn.BCEWithLogitsLoss()

    def loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.criterion(logits.squeeze(), targets)

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.model(inputs)
            probs = torch.sigmoid(logits).squeeze()
        return probs


__all__ = ["FraudLayerParameters", "build_fraud_detection_model", "FraudDetectionHybrid"]
