"""Hybrid fraud detection and kernel module (classical implementation)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn
import numpy as np


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
    """Create a sequential PyTorch model mirroring the layered structure."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


class HybridFraudKernel(nn.Module):
    """
    Hybrid classical fraud‑detection network with RBF kernel.

    The network first extracts features via a multilayer model that mimics a
    photonic circuit.  The extracted features are then compared using a
    Gaussian RBF kernel, enabling downstream kernel‑based learning methods.
    """

    def __init__(
        self,
        fraud_input_params: FraudLayerParameters,
        fraud_layers: Iterable[FraudLayerParameters],
        gamma: float = 1.0,
    ) -> None:
        super().__init__()
        self.fraud_net = build_fraud_detection_program(fraud_input_params, fraud_layers)
        self.gamma = gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return fraud‑detection features."""
        return self.fraud_net(x)

    def rbf_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute Gaussian RBF kernel between two feature vectors."""
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

    def kernel_matrix(
        self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]
    ) -> np.ndarray:
        """Compute Gram matrix between two datasets using extracted features."""
        feats_a = torch.stack([self.fraud_net(x) for x in a])
        feats_b = torch.stack([self.fraud_net(x) for x in b])
        return np.array(
            [
                [self.rbf_kernel(a_i, b_j).item() for b_j in feats_b]
                for a_i in feats_a
            ]
        )


__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "HybridFraudKernel",
]
