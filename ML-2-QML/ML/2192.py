"""
Classical fraud‑detection model with residual layers, optional dropout, and a richer architecture.
"""

from __future__ import annotations

import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable, List, Tuple


@dataclass
class FraudLayerParameters:
    """Parameters describing a fully‑connected layer in the classical model."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]


class FraudDetectionHybrid(nn.Module):
    """
    Classical fraud‑detection model that mirrors the layered structure of the seed
    while adding residual connections and dropout for regularisation.

    Parameters
    ----------
    input_params : FraudLayerParameters
        Parameters for the first layer (treated as the input layer).
    layers : Iterable[FraudLayerParameters]
        Parameters for additional hidden layers.
    dropout : float, optional
        Dropout probability applied after each residual block. Defaults to 0.0.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers: List[nn.Module] = []
        for idx, params in enumerate([input_params] + list(layers)):
            self.layers.append(self._make_layer(params, clip=(idx > 0), dropout=dropout))
        self.output = nn.Linear(2, 1)

    def _make_layer(
        self,
        params: FraudLayerParameters,
        *,
        clip: bool,
        dropout: float,
    ) -> nn.Module:
        """Build a single residual block."""
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
        dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        class Layer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = linear
                self.activation = activation
                self.dropout = dropout_layer
                self.register_buffer("scale", scale)
                self.register_buffer("shift", shift)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out = self.linear(x)
                out = self.activation(out)
                out = out * self.scale + self.shift
                out = self.dropout(out)
                return out

        return Layer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass through all residual blocks."""
        for layer in self.layers:
            x = layer(x)
        return self.output(x)


__all__ = ["FraudLayerParameters", "FraudDetectionHybrid"]
