"""Hybrid fraud detection model combining a classical neural network with a photonic-inspired feature extractor.

The model first applies a stack of parameterized linear layers that mimic the structure of a photonic circuit,
then feeds the resulting features into a small fully‑connected regressor.  This mirrors the classical analogue
of the photonic circuit while adding a learnable post‑processing network, inspired by the EstimatorQNN
regressor.

The class is intentionally lightweight so that it can be used as a drop‑in replacement for the original
FraudDetection model while exposing a richer architecture for research experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

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

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

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
            out = self.activation(self.linear(inputs))
            out = out * self.scale + self.shift
            return out

    return Layer()

class FraudDetectionHybrid(nn.Module):
    """Hybrid fraud‑detection model with a photonic‑style feature extractor
    followed by a small feed‑forward regressor.

    Parameters
    ----------
    input_params : FraudLayerParameters
        Parameters for the first (unclipped) layer.
    layers : Iterable[FraudLayerParameters]
        Parameters for subsequent clipped layers.
    hidden_sizes : Sequence[int], optional
        Sizes of hidden layers in the post‑processing regressor.  Defaults to (8, 4).
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        hidden_sizes: Sequence[int] = (8, 4),
    ) -> None:
        super().__init__()
        self.feature_extractor = nn.Sequential(
            _layer_from_params(input_params, clip=False),
            *(_layer_from_params(l, clip=True) for l in layers),
            nn.Linear(2, 2),
        )
        regressor_layers: List[nn.Module] = []
        in_features = 2
        for size in hidden_sizes:
            regressor_layers.append(nn.Linear(in_features, size))
            regressor_layers.append(nn.Tanh())
            in_features = size
        regressor_layers.append(nn.Linear(in_features, 1))
        self.regressor = nn.Sequential(*regressor_layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Forward pass: feature extraction → regression."""
        features = self.feature_extractor(inputs)
        return self.regressor(features)

__all__ = ["FraudLayerParameters", "FraudDetectionHybrid"]
