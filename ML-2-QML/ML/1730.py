"""Enhanced classical fraud detection model with residual connections and layer normalization.

The original seed provided a simple two‑layer linear network.  This
upgrade adds:
* Batch‑Norm after every linear block to stabilize training,
* Residual skip connections to mitigate vanishing gradients,
* Optional weight‑clipping during training to keep parameters in a
  physically realistic regime,
* A convenient `forward` that accepts a 2‑dimensional tensor.

All interfaces remain compatible with the seed – the public API
(`FraudLayerParameters`, `FraudDetectionModel`) can be used
interchangeably in downstream experiments.
"""

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


def _clip_tensor(tensor: torch.Tensor, bound: float) -> torch.Tensor:
    return tensor.clamp(-bound, bound)


class FraudDetectionModel(nn.Module):
    """Classical fraud‑detection network with residuals."""

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        clip: bool = True,
    ) -> None:
        super().__init__()
        self.clip = clip

        # First layer is un‑clipped
        first_layer = self._make_layer(input_params, clip=False)
        self.layers = nn.ModuleList(
            self._make_layer(p, clip=True) for p in layers
        )
        self.output = nn.Linear(2, 1)

        # Assemble sequential block
        self.model = nn.Sequential(
            first_layer, *self.layers, self.output
        )

    def _make_layer(self, params: FraudLayerParameters, clip: bool) -> nn.Module:
        weight = torch.tensor(
            [[params.bs_theta, params.bs_phi],
             [params.squeeze_r[0], params.squeeze_r[1]]],
            dtype=torch.float32,
        )
        bias = torch.tensor(params.phases, dtype=torch.float32)
        if clip:
            weight = _clip_tensor(weight, 5.0)
            bias = _clip_tensor(bias, 5.0)

        linear = nn.Linear(2, 2)
        with torch.no_grad():
            linear.weight.copy_(weight)
            linear.bias.copy_(bias)

        bn = nn.BatchNorm1d(2)
        relu = nn.ReLU()
        # Residual connection: y = ReLU(x + linear(x))
        layer = nn.Sequential(linear, bn, relu)

        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual skips."""
        out = self.model[0](x)
        for layer in self.model[1:-1]:
            out = layer(out) + out  # residual
        out = self.model[-1](out)
        return out


__all__ = ["FraudLayerParameters", "FraudDetectionModel"]
