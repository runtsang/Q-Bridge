"""Extended classical fraud detection model with dropout, batchnorm, and configurable activations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Callable

import torch
from torch import nn


@dataclass
class FraudLayerParameters:
    """Parameters for a fully‑connected layer in the fraud detection network."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]
    # New hyper‑parameters
    dropout: float = 0.0
    batchnorm: bool = False
    activation: Callable[[torch.Tensor], torch.Tensor] = nn.Tanh()


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def _layer_from_params(
    params: FraudLayerParameters,
    *,
    clip: bool,
    dropout: float,
    use_batchnorm: bool,
    activation: Callable[[torch.Tensor], torch.Tensor],
) -> nn.Module:
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

    modules: list[nn.Module] = [linear]
    if use_batchnorm:
        modules.append(nn.BatchNorm1d(2))
    modules.append(activation)
    if dropout > 0.0:
        modules.append(nn.Dropout(dropout))
    scale = torch.tensor(params.displacement_r, dtype=torch.float32)
    shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(*modules)
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            outputs = self.net(inputs)
            return outputs * self.scale + self.shift

    return Layer()


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    *,
    dropout: float = 0.0,
    use_batchnorm: bool = False,
    activation: Callable[[torch.Tensor], torch.Tensor] = nn.Tanh(),
) -> nn.Sequential:
    """Build a PyTorch fraud‑detection network.

    Parameters
    ----------
    input_params
        Parameters for the first (input) layer.
    layers
        Iterable of parameters for subsequent hidden layers.
    dropout
        Dropout probability applied after every hidden layer.
    use_batchnorm
        If True, a BatchNorm1d layer is inserted after the linear transform.
    activation
        Activation function applied after the linear transform.
    """
    modules: list[nn.Module] = [
        _layer_from_params(
            input_params,
            clip=False,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
            activation=activation,
        )
    ]
    modules.extend(
        _layer_from_params(
            layer,
            clip=True,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
            activation=activation,
        )
        for layer in layers
    )
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


__all__ = ["FraudLayerParameters", "build_fraud_detection_program"]
