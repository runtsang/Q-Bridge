import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Iterable, Tuple


@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


class PhotonicLayer(nn.Module):
    """A lightweight classical layer that emulates the photonic operations
    used in the original fraud‑detection seed."""
    def __init__(self, params: FraudLayerParameters, clip: bool = False):
        super().__init__()
        weight_matrix = torch.tensor(
            [[params.bs_theta, params.bs_phi],
             [params.squeeze_r[0], params.squeeze_r[1]]],
            dtype=torch.float32,
        )
        bias_vector = torch.tensor(params.phases, dtype=torch.float32)
        if clip:
            weight_matrix = weight_matrix.clamp(-5.0, 5.0)
            bias_vector = bias_vector.clamp(-5.0, 5.0)
        self.linear = nn.Linear(2, 2, bias=True)
        with torch.no_grad():
            self.linear.weight.copy_(weight_matrix)
            self.linear.bias.copy_(bias_vector)
        self.activation = nn.Tanh()
        self.scale = nn.Parameter(torch.tensor(params.displacement_r, dtype=torch.float32))
        self.shift = nn.Parameter(torch.tensor(params.displacement_phi, dtype=torch.float32))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out = self.activation(self.linear(inputs))
        out = out * self.scale + self.shift
        return out


class FraudDetectionHybrid(nn.Module):
    """Hybrid classical fraud‑detection model combining a CNN backbone
    with a stack of photonic‑inspired layers."""
    def __init__(self, *layer_params: FraudLayerParameters):
        super().__init__()
        # CNN feature extractor (inspired by Quantum‑NAT)
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_reduce = nn.Linear(16, 2)
        # Photonic layers stack
        self.photonic_layers = nn.ModuleList(
            [PhotonicLayer(layer_params[0], clip=False)] +
            [PhotonicLayer(p, clip=True) for p in layer_params[1:]]
        )
        self.output = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).view(x.shape[0], -1)
        x = self.fc_reduce(x)
        for layer in self.photonic_layers:
            x = layer(x)
        return self.output(x)


__all__ = ["FraudDetectionHybrid", "FraudLayerParameters"]
