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

class ScaleShift(nn.Module):
    """Applies element‑wise scale and shift."""
    def __init__(self, scale: torch.Tensor, shift: torch.Tensor):
        super().__init__()
        self.register_buffer("scale", scale)
        self.register_buffer("shift", shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.shift

class FraudDetectionAdvanced(nn.Module):
    """
    Classical fraud detection model with optional dropout and batch‑norm.
    Each layer is built from FraudLayerParameters, mirroring the photonic
    design but with classical fully‑connected blocks.
    """
    def __init__(self,
                 input_params: FraudLayerParameters,
                 layer_params: List[FraudLayerParameters],
                 dropout: float = 0.0,
                 use_batchnorm: bool = True,
                 clip_bounds: float = 5.0):
        super().__init__()
        self.dropout = dropout
        self.use_batchnorm = use_batchnorm
        self.clip_bounds = clip_bounds

        modules: List[nn.Module] = []

        # first layer without clipping
        modules.append(self._build_layer(input_params, clip=False))

        # subsequent layers with clipping
        for params in layer_params:
            modules.append(self._build_layer(params, clip=True))

        # final output head
        modules.append(nn.Linear(2, 1))

        self.model = nn.Sequential(*modules)

    def _build_layer(self, params: FraudLayerParameters, clip: bool) -> nn.Module:
        weight = torch.tensor([[params.bs_theta, params.bs_phi],
                               [params.squeeze_r[0], params.squeeze_r[1]]],
                              dtype=torch.float32)
        bias = torch.tensor(params.phases, dtype=torch.float32)

        if clip:
            weight = weight.clamp(-self.clip_bounds, self.clip_bounds)
            bias = bias.clamp(-self.clip_bounds, self.clip_bounds)

        linear = nn.Linear(2, 2)
        with torch.no_grad():
            linear.weight.copy_(weight)
            linear.bias.copy_(bias)

        blocks: List[nn.Module] = [linear, nn.Tanh()]

        if self.use_batchnorm:
            blocks.append(nn.BatchNorm1d(2))

        if self.dropout > 0.0:
            blocks.append(nn.Dropout(self.dropout))

        scale = torch.tensor(params.displacement_r, dtype=torch.float32)
        shift = torch.tensor(params.displacement_phi, dtype=torch.float32)
        blocks.append(ScaleShift(scale, shift))

        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

__all__ = ["FraudLayerParameters", "FraudDetectionAdvanced"]
