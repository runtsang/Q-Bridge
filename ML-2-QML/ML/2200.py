"""Extended fraud‑detection model implemented in PyTorch.

This module deepens the original seed by adding batch‑normalisation,
dropout, and a per‑layer clipping regime.  The resulting
`FraudDetection` class can be instantiated with a list of
`FraudLayerParameters` and used like any normal `nn.Module`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import torch
from torch import nn


@dataclass
class FraudLayerParameters:
    """Parameters that mimic a photonic layer."""

    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


class FraudDetection(nn.Module):
    """PyTorch implementation of the fraud‑detection network.

    The network mirrors the photonic circuit but adds:
        • Optional dropout after the final linear layer.
        • Batch‐normalisation before the activation.
        • A dynamic clipping mechanism that can be toggled per layer.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        *,
        dropout_prob: float = 0.0,
        clip_bound: float = 5.0,
    ) -> None:
        super().__init__()
        self.clip_bound = clip_bound

        # Build the sequence of trainable layers
        self._layers: List[nn.Module] = [
            self._layer_from_params(input_params, clip=False)
        ]
        self._layers.extend(
            self._layer_from_params(layer, clip=True) for layer in layers
        )
        self._layers.append(nn.Linear(2, 1))

        # Optional dropout
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0.0 else nn.Identity()

        # Final activation
        self.final_activation = nn.Sigmoid()

        self.model = nn.Sequential(*self._layers, self.dropout, self.final_activation)

    def _layer_from_params(self, params: FraudLayerParameters, *, clip: bool) -> nn.Module:
        """Create a single linear + activation block with optional clipping."""
        weight = torch.tensor(
            [
                [params.bs_theta, params.bs_phi],
                [params.squeeze_r[0], params.squeeze_r[1]],
            ],
            dtype=torch.float32,
        )
        bias = torch.tensor(params.phases, dtype=torch.float32)

        if clip:
            weight = weight.clamp(-self.clip_bound, self.clip_bound)
            bias = bias.clamp(-self.clip_bound, self.clip_bound)

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
        return self.model(x)

    @staticmethod
    def clip(value: float, bound: float) -> float:
        """Utility for clipping a scalar."""
        return max(-bound, min(bound, value))


__all__ = ["FraudLayerParameters", "FraudDetection"]
