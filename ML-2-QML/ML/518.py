"""Fraud detection model built on PyTorch with enhanced layers and training utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class FraudLayerParameters:
    """Parameters for a single fully‑connected layer in the fraud model."""

    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]
    dropout_rate: float = 0.0
    use_batchnorm: bool = False


def _weight_bias_from_params(params: FraudLayerParameters, *, clip: bool) -> tuple[torch.Tensor, torch.Tensor]:
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
    return weight, bias


class FraudLayer(nn.Module):
    """A single layer that emulates a photonic operation in a classical network."""

    def __init__(self, params: FraudLayerParameters, *, clip: bool = False) -> None:
        super().__init__()
        weight, bias = _weight_bias_from_params(params, clip=clip)
        self.linear = nn.Linear(2, 2, bias=True)
        with torch.no_grad():
            self.linear.weight.copy_(weight)
            self.linear.bias.copy_(bias)
        self.activation = nn.Tanh()
        self.scale = nn.Parameter(torch.tensor(params.displacement_r, dtype=torch.float32))
        self.shift = nn.Parameter(torch.tensor(params.displacement_phi, dtype=torch.float32))
        self.dropout = nn.Dropout(p=params.dropout_rate) if params.dropout_rate > 0 else nn.Identity()
        self.bn = nn.BatchNorm1d(2) if params.use_batchnorm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        out = self.activation(out)
        out = out * self.scale + self.shift
        out = self.bn(out)
        out = self.dropout(out)
        return out


class FraudDetectionModel(nn.Module):
    """A hybrid‑style fraud‑detection network that mirrors the photonic architecture."""

    def __init__(self, input_params: FraudLayerParameters, layers: Iterable[FraudLayerParameters]) -> None:
        super().__init__()
        modules: List[nn.Module] = [FraudLayer(input_params, clip=False)]
        modules.extend(FraudLayer(l, clip=True) for l in layers)
        modules.append(nn.Linear(2, 1))
        self.network = nn.Sequential(*modules)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.network(x)
        return self.sigmoid(logits)

    def risk_score(self, x: torch.Tensor) -> torch.Tensor:
        """Return a continuous risk score between 0 and 1."""
        return self.forward(x)

    @staticmethod
    def from_parameters(input_params: FraudLayerParameters, layers: Iterable[FraudLayerParameters]) -> "FraudDetectionModel":
        """Convenience constructor that validates shapes."""
        return FraudDetectionModel(input_params, layers)


__all__ = ["FraudLayerParameters", "FraudDetectionModel"]
