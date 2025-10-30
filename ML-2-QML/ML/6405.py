from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


@dataclass
class FraudLayerParameters:
    """Parameters that mirror the original photonic layer but are now
    interpreted as weights for a small classical feed‑forward network."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


class FraudDetectionHybrid(nn.Module):
    """Classical fraud detection model with a gating network.

    The architecture consists of:
      - A feature‑wise gating network that learns to mask the raw input.
      - A main MLP that predicts the fraud probability.
    """
    def __init__(self, gating_hidden: int = 8, main_hidden: int = 2):
        super().__init__()
        # Feature‑wise gating network
        self.gate = nn.Sequential(
            nn.Linear(2, gating_hidden),
            nn.ReLU(),
            nn.Linear(gating_hidden, 2),
            nn.Sigmoid()
        )
        # Main classifier
        self.main = nn.Sequential(
            nn.Linear(2, main_hidden),
            nn.ReLU(),
            nn.Linear(main_hidden, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        mask = self.gate(x)
        gated_x = x * mask
        return self.main(gated_x)

    @staticmethod
    def from_params(
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
    ) -> "FraudDetectionHybrid":
        """Create a model from the photonic parameters.  The parameters
        are used only to initialise a simple linear layer that mimics
        the first photonic layer."""
        weight = torch.tensor(
            [
                [input_params.bs_theta, input_params.bs_phi],
                [input_params.squeeze_r[0], input_params.squeeze_r[1]],
            ],
            dtype=torch.float32
        )
        bias = torch.tensor(input_params.phases, dtype=torch.float32)
        model = FraudDetectionHybrid()
        # Overwrite the first linear layer of the main network
        model.main[0].weight.data.copy_(weight.t())
        model.main[0].bias.data.copy_(bias)
        return model


__all__ = ["FraudLayerParameters", "FraudDetectionHybrid"]
