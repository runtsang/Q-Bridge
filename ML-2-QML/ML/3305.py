"""Hybrid classical kernel with fraud‑detection style feature
transformation and radial‑basis evaluation."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from dataclasses import dataclass
from typing import Sequence, Iterable

@dataclass
class FraudLayerParameters:
    """Parameters describing a fully‑connected fraud‑detection layer."""
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
        [[params.bs_theta, params.bs_phi],
         [params.squeeze_r[0], params.squeeze_r[1]]],
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


class HybridKernelFraudDetection(nn.Module):
    """Classical kernel that transforms inputs via a fraud‑detection
    network and then evaluates an RBF on the transformed data."""
    def __init__(
        self,
        gamma: float = 1.0,
        fraud_params: list[FraudLayerParameters] | None = None,
        clip: bool = True,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.fraud_net = nn.Sequential()
        if fraud_params:
            layers = [_layer_from_params(fraud_params[0], clip=False)]
            layers += [_layer_from_params(p, clip=clip) for p in fraud_params[1:]]
            layers.append(nn.Linear(2, 1))
            self.fraud_net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        tx = self.fraud_net(x) if self.fraud_net else x
        ty = self.fraud_net(y) if self.fraud_net else y
        diff = tx - ty
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

def kernel_matrix(
    a: Sequence[torch.Tensor],
    b: Sequence[torch.Tensor],
    gamma: float = 1.0,
    fraud_params: list[FraudLayerParameters] | None = None,
) -> np.ndarray:
    model = HybridKernelFraudDetection(gamma, fraud_params)
    return np.array([[model(x, y).item() for y in b] for x in a])

__all__ = ["FraudLayerParameters", "HybridKernelFraudDetection", "kernel_matrix"]
