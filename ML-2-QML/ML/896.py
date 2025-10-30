"""Enhanced classical fraud detection model with dropout, batch normalization, and gradient utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, List

import torch
from torch import nn
import torch.nn.functional as F


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
    dropout: float = 0.0  # new field for dropout probability


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
            self.batchnorm = nn.BatchNorm1d(2)
            self.dropout = nn.Dropout(params.dropout)
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            outputs = self.linear(inputs)
            outputs = self.batchnorm(outputs)
            outputs = self.activation(outputs)
            outputs = outputs * self.scale + self.shift
            outputs = self.dropout(outputs)
            return outputs

    return Layer()


class FraudDetectionModel(nn.Module):
    """Complete fraud detection neural network with optional dropout and batch normalization."""

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        *,
        final_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        modules: List[nn.Module] = [_layer_from_params(input_params, clip=False)]
        modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
        modules.append(nn.Linear(2, 1))
        modules.append(nn.Sigmoid())
        if final_dropout > 0.0:
            modules.append(nn.Dropout(final_dropout))
        self.model = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def jacobian(self, x: torch.Tensor) -> torch.Tensor:
        """Return the Jacobian of the output w.r.t. the input."""
        return torch.autograd.functional.jacobian(self, x)

    def summary(self) -> str:
        """Return a human‑readable summary of the network architecture."""
        return str(self.model)

    def to_torchscript(self) -> torch.jit.ScriptModule:
        """Compile the model to TorchScript for deployment."""
        return torch.jit.script(self.model)


__all__ = ["FraudLayerParameters", "FraudDetectionModel"]
