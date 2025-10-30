"""Enhanced classical fraud‑detection model with residual connections and optional feature scaling."""

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


def _clip(value: float, bound: float) -> float:
    """Clamp values to a safe range for numerical stability."""
    return max(-bound, min(bound, value))


def _build_layer(
    params: FraudLayerParameters,
    *,
    clip: bool,
    residual: bool,
) -> nn.Module:
    """Return a single residual‑style block that can be used in a Sequential."""
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
            if residual and inputs.shape == out.shape:
                out = out + inputs
            return out

    return Layer()


class FraudDetectionEnhanced:
    """Encapsulates the enhanced fraud‑detection neural network."""

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        *,
        residual: bool = True,
        clip: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        input_params
            Parameters for the first (input) layer.
        layers
            Iterable of parameters for subsequent layers.
        residual
            Whether to enable residual connections in each block.
        clip
            Whether to clip weight and bias values for numerical stability.
        """
        self.input_params = input_params
        self.layers = list(layers)
        self.residual = residual
        self.clip = clip
        self.model = self._build_model()

    def _build_model(self) -> nn.Sequential:
        """Build the sequential PyTorch model."""
        modules: list[nn.Module] = [
            _build_layer(self.input_params, clip=False, residual=self.residual)
        ]
        modules += [
            _build_layer(layer, clip=self.clip, residual=self.residual)
            for layer in self.layers
        ]
        modules.append(nn.Linear(2, 1))
        return nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.model(x)

    def parameters(self):
        """Return the parameters of the underlying PyTorch model."""
        return self.model.parameters()


__all__ = ["FraudDetectionEnhanced", "FraudLayerParameters"]
