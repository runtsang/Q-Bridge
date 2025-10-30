"""
FraudDetectionModel – classical residual MLP with dropout and batch norm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class FraudLayerParameters:
    """Parameters describing a fully connected layer in the classical model.

    Mirrors the photonic parameters but interpreted as linear layer weights
    and biases.  ``bs_theta`` and ``bs_phi`` become the main weight matrix,
    ``phases`` become the bias, and the remaining tuples are used to scale
    and shift the activations (acting as a learned affine transform).
    """
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


class _FraudLayer(nn.Module):
    """Internal building block: a linear layer followed by tanh, scaling and shifting.

    The layer weight is constructed from ``bs_theta`` and ``bs_phi`` and is
    optionally clipped to promote numerical stability.  Dropout and batch‑norm
    are added in the high‑level model for regularisation.
    """
    def __init__(self, params: FraudLayerParameters, clip: bool = False, dropout: float | None = None):
        super().__init__()
        weight = torch.tensor([[params.bs_theta, params.bs_phi],
                               [params.squeeze_r[0], params.squeeze_r[1]]],
                              dtype=torch.float32)
        bias = torch.tensor(params.phases, dtype=torch.float32)
        if clip:
            weight = weight.clamp(-5.0, 5.0)
            bias = bias.clamp(-5.0, 5.0)
        self.linear = nn.Linear(2, 2)
        with torch.no_grad():
            self.linear.weight.copy_(weight)
            self.linear.bias.copy_(bias)
        self.activation = nn.Tanh()
        self.scale = nn.Parameter(torch.tensor(params.displacement_r, dtype=torch.float32))
        self.shift = nn.Parameter(torch.tensor(params.displacement_phi, dtype=torch.float32))
        self.dropout = nn.Dropout(dropout) if dropout is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(self.linear(x))
        out = out * self.scale + self.shift
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class FraudDetectionModel(nn.Module):
    """Residual MLP for fraud‑detection.

    Parameters
    ----------
    input_params : FraudLayerParameters
        Parameters for the first (non‑clipped) layer.
    layers : Iterable[FraudLayerParameters]
        Subsequent layers, each clipped to keep the model bounded.
    dropout : float, optional
        Dropout probability applied after each internal layer.
    """
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        dropout: float | None = 0.1,
    ) -> None:
        super().__init__()
        self.layers: List[nn.Module] = []
        self.layers.append(_FraudLayer(input_params, clip=False, dropout=dropout))
        self.layers.extend(
            _FraudLayer(l, clip=True, dropout=dropout) for l in layers
        )
        self.final = nn.Linear(2, 1)
        self.batch_norm = nn.BatchNorm1d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        for layer in self.layers:
            out = layer(residual)
            residual = out  # residual connection
        out = self.batch_norm(residual)
        out = self.final(out)
        return torch.sigmoid(out)

__all__ = ["FraudLayerParameters", "FraudDetectionModel"]
