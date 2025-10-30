"""Enhanced fraud detection model with dropout and parameter clipping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn

@dataclass
class FraudLayerParams:
    """Parameters for a single classical fraud detection layer."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

class FraudDetectionModel(nn.Module):
    """A PyTorch model that mirrors the photonic fraud detection architecture.

    The model supports an arbitrary number of layers, optional dropout, and
    automatic clipping of weights/biases for the hidden layers to keep the
    parameters within a physically realistic range.
    """
    def __init__(
        self,
        input_params: FraudLayerParams,
        layers: Iterable[FraudLayerParams],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(self._layer_from_params(input_params, clip=False))
        for lp in layers:
            self.layers.append(self._layer_from_params(lp, clip=True))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.output_layer = nn.Linear(2, 1)

    def _layer_from_params(self, params: FraudLayerParams, *, clip: bool) -> nn.Module:
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
                out = self.activation(self.linear(inputs))
                out = out * self.scale + self.shift
                return out

        return Layer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        x = self.dropout(x)
        return self.output_layer(x)

def build_fraud_detection_model(
    input_params: FraudLayerParams,
    layers: Iterable[FraudLayerParams],
    dropout: float = 0.0,
) -> FraudDetectionModel:
    """Convenience factory that returns a fully‑initialised model."""
    return FraudDetectionModel(input_params, layers, dropout=dropout)

__all__ = ["FraudLayerParams", "FraudDetectionModel", "build_fraud_detection_model"]
