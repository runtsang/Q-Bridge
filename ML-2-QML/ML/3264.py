"""Hybrid classical model combining convolutional filtering and fraud‑detection style layers."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Iterable


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


class _FraudLayer(nn.Module):
    def __init__(self, params: FraudLayerParameters, clip: bool = False):
        super().__init__()
        weight = torch.tensor(
            [[params.bs_theta, params.bs_phi],
             [params.squeeze_r[0], params.squeeze_r[1]]],
            dtype=torch.float32,
        )
        bias = torch.tensor(params.phases, dtype=torch.float32)
        if clip:
            weight = weight.clamp(-5.0, 5.0)
            bias = bias.clamp(-5.0, 5.0)
        self.linear = nn.Linear(2, 2)
        with torch.no_grad():
            self.linear.weight.copy_(weight)
            self.linear.bias.copy_(bias)
        self.activation = nn.Tanh()
        self.scale = nn.Parameter(
            torch.tensor(params.displacement_r, dtype=torch.float32)
        )
        self.shift = nn.Parameter(
            torch.tensor(params.displacement_phi, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(self.linear(x))
        out = out * self.scale + self.shift
        return out


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    modules = [_FraudLayer(input_params, clip=False)]
    modules += [_FraudLayer(l, clip=True) for l in layers]
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


class HybridQuanvolution(nn.Module):
    """Classical hybrid model: convolutional feature extractor followed by fraud‑detection style head."""

    def __init__(
        self,
        n_filters: int = 4,
        kernel_size: int = 2,
        stride: int = 2,
        head_layers: int = 3,
    ):
        super().__init__()
        self.conv = nn.Conv2d(1, n_filters, kernel_size=kernel_size, stride=stride)
        # Construct head with random parameters for demonstration
        rng = torch.randn
        input_params = FraudLayerParameters(
            bs_theta=rng(1).item(),
            bs_phi=rng(1).item(),
            phases=(rng(1).item(), rng(1).item()),
            squeeze_r=(rng(1).item(), rng(1).item()),
            squeeze_phi=(rng(1).item(), rng(1).item()),
            displacement_r=(rng(1).item(), rng(1).item()),
            displacement_phi=(rng(1).item(), rng(1).item()),
            kerr=(rng(1).item(), rng(1).item()),
        )
        layers = [
            FraudLayerParameters(
                bs_theta=rng(1).item(),
                bs_phi=rng(1).item(),
                phases=(rng(1).item(), rng(1).item()),
                squeeze_r=(rng(1).item(), rng(1).item()),
                squeeze_phi=(rng(1).item(), rng(1).item()),
                displacement_r=(rng(1).item(), rng(1).item()),
                displacement_phi=(rng(1).item(), rng(1).item()),
                kerr=(rng(1).item(), rng(1).item()),
            )
            for _ in range(head_layers - 1)
        ]
        self.head = build_fraud_detection_program(input_params, layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        flat = features.view(features.size(0), -1)
        logits = self.head(flat)
        return F.log_softmax(logits, dim=-1)


__all__ = ["HybridQuanvolution", "FraudLayerParameters", "build_fraud_detection_program"]
