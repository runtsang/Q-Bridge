"""Enhanced classical fraud detection model with attention and dual‑head output."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class FraudLayerParameters:
    """Parameters describing a fully connected layer in the classical model."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]


def _clip(value: float, bound: float) -> float:
    """Clamp value to the range [-bound, bound]."""
    return max(-bound, min(bound, value))


class MultiHeadAttention(nn.Module):
    """Simple 1‑head self‑attention block used in the fraud model."""
    def __init__(self, dim: int, attn_dim: int = 4):
        super().__init__()
        self.query = nn.Linear(dim, attn_dim, bias=False)
        self.key   = nn.Linear(dim, attn_dim, bias=False)
        self.value = nn.Linear(dim, attn_dim, bias=False)
        self.out   = nn.Linear(attn_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (K.size(-1) ** 0.5)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        return self.out(out)


class FraudDetectionEnhanced(nn.Module):
    """
    Build a fraud detection network from a list of `FraudLayerParameters`.

    Parameters
    ----------
    input_params : FraudLayerParameters
        Parameters for the first layer (no clipping).
    layers : Iterable[FraudLayerParameters]
        Parameters for subsequent layers (clipped to a safe range).
    use_attention : bool, optional
        If ``True`` inserts a simple self‑attention block after each
        linear+activation+scale‑shift block.
    dual_head : bool, optional
        If ``True`` adds an auxiliary binary output head.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        *,
        use_attention: bool = False,
        dual_head: bool = False,
    ):
        super().__init__()
        self.use_attention = use_attention
        self.dual_head = dual_head

        body_modules: list[nn.Module] = []

        # first layer, no clipping
        body_modules.append(self._layer_from_params(input_params, clip=False))

        # subsequent layers
        for params in layers:
            body_modules.append(self._layer_from_params(params, clip=True))

        self.feature_extractor = nn.Sequential(*body_modules)
        self.class_head = nn.Linear(2, 1)

        if dual_head:
            self.aux_head = nn.Linear(2, 1)

    def _layer_from_params(
        self, params: FraudLayerParameters, *, clip: bool
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

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                out = self.activation(self.linear(inputs))
                out = out * self.scale + self.shift
                return out

        layer = Layer()
        if self.use_attention:
            return nn.Sequential(layer, MultiHeadAttention(2))
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(x)
        out = self.class_head(features)
        if self.dual_head:
            aux = self.aux_head(features)
            return out, aux
        return out


__all__ = ["FraudDetectionEnhanced", "FraudLayerParameters"]
