"""Extended classical fraud‑detection model with dropout and training helper."""

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


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def _layer_from_params(
    params: FraudLayerParameters,
    *,
    clip: bool,
    dropout_rate: float,
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
            self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            outputs = self.activation(self.linear(inputs))
            outputs = self.dropout(outputs)
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    *,
    dropout_rate: float = 0.0,
) -> nn.Sequential:
    """Create a sequential PyTorch model mirroring the layered structure."""
    modules = [_layer_from_params(input_params, clip=False, dropout_rate=dropout_rate)]
    modules.extend(
        _layer_from_params(layer, clip=True, dropout_rate=dropout_rate)
        for layer in layers
    )
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


class FraudDetectionModel:
    """Convenient wrapper around the sequential fraud‑detection network."""

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        *,
        dropout_rate: float = 0.0,
    ) -> None:
        self.model = build_fraud_detection_program(
            input_params, layers, dropout_rate=dropout_rate
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def train_on(
        self,
        data_loader,
        lr: float = 1e-3,
        epochs: int = 10,
        weight_decay: float = 1e-4,
    ) -> None:
        """Simple training loop using BCEWithLogitsLoss."""
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        criterion = nn.BCEWithLogitsLoss()
        self.model.train()
        for epoch in range(epochs):
            for batch_x, batch_y in data_loader:
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits.squeeze(), batch_y.float())
                loss.backward()
                optimizer.step()


__all__ = ["FraudDetectionParameters", "FraudDetectionModel"]
