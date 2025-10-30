"""Hybrid fraud detection model – classical implementation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn


@dataclass
class FraudLayerParameters:
    """Parameters describing a fully‑connected layer in the classical model."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


class FraudDetectionModel:
    """Classical fraud‑detection model with optional regularisation."""
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        *,
        drop_prob: float = 0.0,
        use_batchnorm: bool = False,
    ) -> None:
        self.input_params = input_params
        self.layers = list(layers)
        self.drop_prob = drop_prob
        self.use_batchnorm = use_batchnorm

    def _clip(self, value: float, bound: float) -> float:
        return max(-bound, min(bound, value))

    def _layer_from_params(self, params: FraudLayerParameters, clip: bool) -> nn.Module:
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
                self.dropout = nn.Dropout(self.drop_prob) if self.drop_prob > 0 else nn.Identity()
                self.bn = nn.BatchNorm1d(2) if self.use_batchnorm else nn.Identity()

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
                x = self.linear(inputs)
                x = self.activation(x)
                x = self.bn(x)
                x = self.dropout(x)
                x = x * self.scale + self.shift
                return x

        return Layer()

    def build_classical(self) -> nn.Sequential:
        """Construct the full PyTorch model."""
        modules = [self._layer_from_params(self.input_params, clip=False)]
        modules.extend(self._layer_from_params(l, clip=True) for l in self.layers)
        modules.append(nn.Linear(2, 1))
        return nn.Sequential(*modules)

    def train_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn,
        data: torch.Tensor,
        target: torch.Tensor,
    ) -> float:
        """One training step for the classical network."""
        model.train()
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        return loss.item()


__all__ = ["FraudLayerParameters", "FraudDetectionModel"]
