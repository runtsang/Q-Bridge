"""Classical fraud‑detection model with enhanced regularisation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import nn


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
    """Clip a scalar to the interval [-bound, bound]."""
    return max(-bound, min(bound, value))


def _layer_from_params(
    params: FraudLayerParameters,
    *,
    clip: bool = True,
    dropout_rate: float = 0.0,
) -> nn.Module:
    """Build a single linear‑activation block with optional dropout."""
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

    block = nn.Sequential(
        linear,
        activation,
        nn.BatchNorm1d(2),
        nn.Dropout(dropout_rate),
    )
    # Register scale/shift as buffers to avoid gradients
    block.register_buffer("scale", scale)
    block.register_buffer("shift", shift)

    def forward(x: torch.Tensor) -> torch.Tensor:
        out = block(x)
        return out * block.scale + block.shift

    return nn.ModuleList([block, nn.Module()]).__class__(forward)


class FraudDetectionModel:
    """Enhanced classical fraud‑detection pipeline."""

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        dropout_rate: float = 0.1,
    ) -> None:
        self.input_params = input_params
        self.layers = list(layers)
        self.dropout_rate = dropout_rate
        self.model = self._build_model()

    def _build_model(self) -> nn.Sequential:
        modules = [_layer_from_params(self.input_params, clip=False)]
        modules.extend(
            _layer_from_params(l, clip=True, dropout_rate=self.dropout_rate)
            for l in self.layers
        )
        modules.append(nn.Linear(2, 1))
        return nn.Sequential(*modules)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning a raw score."""
        return self.model(x)

    def evaluate(self, dataloader) -> Tuple[float, float]:
        """Compute mean squared error and accuracy on a dataset."""
        self.model.eval()
        mse = torch.nn.functional.mse_loss
        correct, total = 0, 0
        loss_sum = 0.0
        with torch.no_grad():
            for xb, yb in dataloader:
                out = self.predict(xb).squeeze()
                loss_sum += mse(out, yb).item() * xb.size(0)
                preds = (out > 0.5).float()
                correct += (preds == yb).sum().item()
                total += xb.size(0)
        return loss_sum / total, correct / total

    def __repr__(self) -> str:
        return f"<FraudDetectionModel layers={len(self.layers)}>"


__all__ = ["FraudLayerParameters", "FraudDetectionModel"]
