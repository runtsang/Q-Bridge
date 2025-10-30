"""Extended classical fraud detection model with residual blocks and dropout."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

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


class FraudDetectionModel(nn.Module):
    """Extended classical fraud detection model with residual blocks, batch norm, and dropout.

    Each layer corresponds to one photonic layer described by ``FraudLayerParameters``.
    """

    def __init__(
        self,
        params_list: List[FraudLayerParameters],
        dropout: float = 0.3,
        activation: nn.Module = nn.ReLU(),
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for params in params_list:
            linear = nn.Linear(2, 2)
            weight = torch.tensor(
                [[params.bs_theta, params.bs_phi],
                 [params.squeeze_r[0], params.squeeze_r[1]]],
                dtype=torch.float32,
            )
            bias = torch.tensor(params.phases, dtype=torch.float32)
            linear.weight.data.copy_(weight)
            linear.bias.data.copy_(bias)
            self.layers.append(
                nn.Sequential(
                    linear,
                    nn.BatchNorm1d(2),
                    activation,
                    nn.Dropout(dropout),
                )
            )
        self.final = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.layers:
            residual = out
            out = layer(out)
            out = out + residual  # residual connection
        return torch.sigmoid(self.final(out))
