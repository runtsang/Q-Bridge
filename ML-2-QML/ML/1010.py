"""Robust classical fraud detection model with residual connections and training utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import torch
from torch import nn, Tensor
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


def _clip(value: float, bound: float) -> float:
    """Clamp a scalar to the range [-bound, bound]."""
    return max(-bound, min(bound, value))


def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
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

        def forward(self, inputs: Tensor) -> Tensor:  # type: ignore[override]
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()


class FraudDetection(nn.Module):
    """Classical fraud‑detection neural network with optional residual connections."""

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        use_residual: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.use_residual = use_residual
        self.layers: List[nn.Module] = [
            _layer_from_params(input_params, clip=False)
        ]
        self.layers.extend(_layer_from_params(layer, clip=True) for layer in layers)
        self.dropout = nn.Dropout(dropout)
        self.linear_out = nn.Linear(2, 1)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x: Tensor) -> Tensor:
        out = x
        for layer in self.layers:
            residual = out
            out = layer(out)
            if self.use_residual:
                out = out + residual
            out = self.dropout(out)
        out = self.linear_out(out)
        return out

    def predict(self, x: Tensor) -> Tensor:
        """Return probability predictions."""
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def compute_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        return self.loss_fn(logits, targets)

    def step(self, optimizer: torch.optim.Optimizer, x: Tensor, y: Tensor) -> Tensor:
        optimizer.zero_grad()
        logits = self.forward(x)
        loss = self.compute_loss(logits, y)
        loss.backward()
        optimizer.step()
        return loss

    @staticmethod
    def from_parameters(
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        **kwargs,
    ) -> "FraudDetection":
        """Convenience constructor."""
        return FraudDetection(input_params, layers, **kwargs)

    def get_parameters(self) -> List[FraudLayerParameters]:
        """Return the parameters of each layer as dataclass instances."""
        params = []
        for layer in self.layers:
            weight = layer.linear.weight.detach().cpu().numpy()
            bias = layer.linear.bias.detach().cpu().numpy()
            scale = layer.scale.detach().cpu().numpy()
            shift = layer.shift.detach().cpu().numpy()
            params.append(
                FraudLayerParameters(
                    bs_theta=weight[0, 0],
                    bs_phi=weight[0, 1],
                    phases=(bias[0], bias[1]),
                    squeeze_r=(weight[1, 0], weight[1, 1]),
                    squeeze_phi=(scale[0], scale[1]),
                    displacement_r=(scale[0], scale[1]),
                    displacement_phi=(shift[0], shift[1]),
                    kerr=(0.0, 0.0),  # not used in this implementation
                )
            )
        return params


__all__ = ["FraudLayerParameters", "FraudDetection"]
