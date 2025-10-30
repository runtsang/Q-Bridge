"""Classical fraud detection model with auxiliary risk‑score head.

The implementation builds on the original two‑layer architecture
but introduces an auxiliary regression head that predicts a
continuous risk score.  The model is fully differentiable
and can be trained with standard optimizers.  The API
mirrors the original seed while adding a `compute_loss`
method that returns a weighted sum of binary cross‑entropy
and mean‑squared‑error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn

@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

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
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

class FraudDetectionModel(nn.Module):
    """Classic fraud detector with auxiliary risk‑score head.

    Parameters
    ----------
    input_params : FraudLayerParameters
        Parameters for the first (unclipped) layer.
    hidden_params : Iterable[FraudLayerParameters]
        Parameters for subsequent clipped layers.
    aux_hidden_dim : int, optional
        Hidden dimension of the auxiliary regression head.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        hidden_params: Iterable[FraudLayerParameters],
        aux_hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.fraud_net = build_fraud_detection_program(input_params, hidden_params)
        self.aux_head = nn.Sequential(
            nn.Linear(2, aux_hidden_dim),
            nn.ReLU(),
            nn.Linear(aux_hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return fraud logits and risk‑score prediction."""
        logits = self.fraud_net(x)
        risk_score = self.aux_head(logits)
        return logits, risk_score

    def compute_loss(
        self,
        logits: torch.Tensor,
        risk_score: torch.Tensor,
        labels: torch.Tensor,
        risk_targets: torch.Tensor,
        fraud_weight: float = 1.0,
        risk_weight: float = 0.5,
    ) -> torch.Tensor:
        """Weighted sum of binary cross‑entropy and MSE."""
        bce = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        mse = nn.functional.mse_loss(risk_score, risk_targets)
        return fraud_weight * bce + risk_weight * mse

__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "FraudDetectionModel",
]
