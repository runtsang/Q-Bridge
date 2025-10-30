"""Classical fraud detection model with shared parameterization and advanced regularization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F


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


class FraudDetectionHybrid(nn.Module):
    """Hybrid fraud detection model that maps 2‑dimensional input to a binary label.
    The architecture is a stack of fully‑connected layers with skip connections and
    dropout, followed by a final linear classifier.  The same FraudLayerParameters
    dataclass is used to keep the parameter space coherent with the quantum side."""
    def __init__(self,
                 input_params: FraudLayerParameters,
                 hidden_params: Iterable[FraudLayerParameters],
                 dropout: float = 0.2) -> None:
        super().__init__()
        self.input_layer = self._layer_from_params(input_params, clip=False)
        self.hidden_layers = nn.ModuleList(
            [self._layer_from_params(p, clip=True) for p in hidden_params]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2, 1)

    @staticmethod
    def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
        weight = torch.tensor([[params.bs_theta, params.bs_phi],
                               [params.squeeze_r[0], params.squeeze_r[1]]],
                              dtype=torch.float32)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.input_layer(x)
        for layer in self.hidden_layers:
            out = F.relu(layer(out) + out)  # skip connection
        out = self.dropout(out)
        out = self.classifier(out)
        return torch.sigmoid(out)

__all__ = ["FraudLayerParameters", "FraudDetectionHybrid"]
