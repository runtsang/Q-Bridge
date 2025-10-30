from __future__ import annotations

import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable, List

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

def _layer_from_params(params: FraudLayerParameters, clip: bool) -> nn.Module:
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
            out = self.activation(self.linear(inputs))
            out = out * self.scale + self.shift
            return out

    return Layer()

class QCNNHybridModel(nn.Module):
    """Hybrid QCNN that alternates convolution and pooling steps.

    The architecture is:
        conv → pool → conv → pool → conv → pool → linear → sigmoid
    Each conv/pool block is a 2‑to‑2 linear layer with photonic‑style parameters.
    """

    def __init__(
        self,
        conv_params: Iterable[FraudLayerParameters],
        pool_params: Iterable[FraudLayerParameters],
    ) -> None:
        super().__init__()
        conv_modules: List[nn.Module] = [_layer_from_params(p, clip=False) for p in conv_params]
        pool_modules: List[nn.Module] = [_layer_from_params(p, clip=True) for p in pool_params]
        self.features = nn.Sequential(
            *conv_modules,
            *pool_modules,
            nn.Linear(2, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out = self.features(x)
        return self.sigmoid(out)

def QCNN() -> QCNNHybridModel:
    """Factory that builds a QCNNHybridModel from photonic parameters."""
    conv_params = [
        FraudLayerParameters(
            bs_theta=0.1, bs_phi=0.2,
            phases=(0.3, 0.4),
            squeeze_r=(0.5, 0.6),
            squeeze_phi=(0.7, 0.8),
            displacement_r=(0.9, 1.0),
            displacement_phi=(1.1, 1.2),
            kerr=(0.0, 0.0),
        ),
        FraudLayerParameters(
            bs_theta=0.15, bs_phi=0.25,
            phases=(0.35, 0.45),
            squeeze_r=(0.55, 0.65),
            squeeze_phi=(0.75, 0.85),
            displacement_r=(0.95, 1.05),
            displacement_phi=(1.15, 1.25),
            kerr=(0.0, 0.0),
        ),
    ]
    pool_params = [
        FraudLayerParameters(
            bs_theta=0.2, bs_phi=0.3,
            phases=(0.4, 0.5),
            squeeze_r=(0.6, 0.7),
            squeeze_phi=(0.8, 0.9),
            displacement_r=(1.0, 1.1),
            displacement_phi=(1.2, 1.3),
            kerr=(0.0, 0.0),
        ),
        FraudLayerParameters(
            bs_theta=0.25, bs_phi=0.35,
            phases=(0.45, 0.55),
            squeeze_r=(0.65, 0.75),
            squeeze_phi=(0.85, 0.95),
            displacement_r=(1.05, 1.15),
            displacement_phi=(1.25, 1.35),
            kerr=(0.0, 0.0),
        ),
    ]
    return QCNNHybridModel(conv_params, pool_params)

__all__ = ["FraudLayerParameters", "QCNNHybridModel", "QCNN"]
