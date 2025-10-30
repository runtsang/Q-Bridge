"""Enhanced classical fraud detection model using PyTorch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn


@dataclass
class FraudLayerParameters:
    """Describes a single fully‑connected layer in the classical model."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


def _clip(value: float, bound: float) -> float:
    """Clip a scalar to the range [-bound, bound]."""
    return max(-bound, min(bound, value))


def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    """Create a single linear‑activation‑scale block from the parameters."""
    weight = torch.tensor(
        [[params.bs_theta, params.bs_phi], [params.squeeze_r[0], params.squeeze_r[1]]],
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


class FraudDetectionEnhanced:
    """
    A PyTorch model that mirrors the layered photonic architecture.

    Parameters
    ----------
    input_params : FraudLayerParameters
        Parameters for the initial (non‑clipped) layer.
    layers : Iterable[FraudLayerParameters]
        Parameters for the subsequent layers; each will have its weights clipped.
    dropout : float, optional
        Dropout probability applied after each hidden layer.
    weight_decay : float, optional
        L2 regularisation coefficient for optimizer.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        *,
        dropout: float = 0.2,
        weight_decay: float = 1e-4,
    ) -> None:
        self.input_params = input_params
        self.layers = list(layers)
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.model = self._build_model()

    def _build_model(self) -> nn.Sequential:
        modules: Sequence[nn.Module] = [
            _layer_from_params(self.input_params, clip=False),
            nn.Dropout(self.dropout),
        ]
        for layer in self.layers:
            modules.extend(
                [
                    _layer_from_params(layer, clip=True),
                    nn.BatchNorm1d(2),
                    nn.Dropout(self.dropout),
                ]
            )
        modules.extend([nn.Linear(2, 1), nn.Sigmoid()])
        return nn.Sequential(*modules)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Standard forward pass."""
        return self.model(inputs)

    def get_weight_decay(self) -> float:
        """Return the L2 regularisation coefficient for use in an optimiser."""
        return self.weight_decay


__all__ = ["FraudLayerParameters", "FraudDetectionEnhanced"]
