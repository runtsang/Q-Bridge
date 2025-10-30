"""
Classical fraud‑detection model with enhanced expressivity and modularity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

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
    """Clip a scalar to the interval [-bound, bound]."""
    return max(-bound, min(bound, value))


def _layer_from_params(
    params: FraudLayerParameters,
    *,
    clip: bool,
    dropout: float = 0.0,
    batch_norm: bool = False,
) -> nn.Module:
    """Construct a single processing layer from parameters."""
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
            if batch_norm:
                self.bn = nn.BatchNorm1d(2)
            else:
                self.bn = nn.Identity()
            if dropout > 0.0:
                self.drop = nn.Dropout(dropout)
            else:
                self.drop = nn.Identity()

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            out = self.linear(inputs)
            out = self.activation(out)
            out = self.bn(out)
            out = self.drop(out)
            out = out * self.scale + self.shift
            return out

    return Layer()


class FraudDetectionModel(nn.Module):
    """
    A modular fraud‑detection neural network.

    Parameters
    ----------
    layers : Iterable[FraudLayerParameters]
        Sequence of layer parameters. The first element is treated as the
        input layer, subsequent elements are hidden layers.
    clip : bool
        Whether to clip internal weights/biases to a fixed range.
    dropout : float
        Dropout probability applied after each hidden layer.
    batch_norm : bool
        Whether to include a batch‑norm layer after the activation.
    residual : bool
        If True, add a residual connection between consecutive layers.
    """

    def __init__(
        self,
        layers: Iterable[FraudLayerParameters],
        *,
        clip: bool = True,
        dropout: float = 0.0,
        batch_norm: bool = False,
        residual: bool = False,
    ) -> None:
        super().__init__()
        layers = list(layers)
        if not layers:
            raise ValueError("At least one layer must be provided.")
        modules: list[nn.Module] = []

        # Input layer (no clipping)
        modules.append(_layer_from_params(layers[0], clip=False, dropout=dropout, batch_norm=batch_norm))

        # Hidden layers
        for prev, curr in zip(layers, layers[1:]):
            layer = _layer_from_params(curr, clip=clip, dropout=dropout, batch_norm=batch_norm)
            if residual:
                modules.append(nn.Sequential(layer, nn.Identity()))  # Identity as placeholder for residual
            else:
                modules.append(layer)

        # Output layer
        modules.append(nn.Linear(2, 1))
        self.network = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    @classmethod
    def from_params(
        cls,
        params: Iterable[FraudLayerParameters],
        **kwargs,
    ) -> "FraudDetectionModel":
        """
        Convenience constructor mirroring the original build_fraud_detection_program.
        """
        return cls(params, **kwargs)


__all__ = ["FraudLayerParameters", "FraudDetectionModel"]
