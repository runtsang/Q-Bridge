import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Iterable, List, Optional

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

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()

class FraudDetectionHybridModel(nn.Module):
    """
    Classical hybrid fraud detection model.
    Combines a 2×2 patch convolution filter (mimicking a quantum kernel)
    with a stack of photonic-inspired fully‑connected layers.
    """

    def __init__(
        self,
        conv_out_channels: int = 4,
        conv_kernel: int = 2,
        conv_stride: int = 2,
        fraud_params: Optional[List[FraudLayerParameters]] = None,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=conv_out_channels,
            kernel_size=conv_kernel,
            stride=conv_stride,
        )
        self.flatten = nn.Flatten()
        self.reduction = nn.Linear(conv_out_channels * 14 * 14, 2)
        self.fraud_layers = nn.Sequential()
        if fraud_params:
            for i, params in enumerate(fraud_params):
                self.fraud_layers.add_module(f"layer_{i}", _layer_from_params(params, clip=(i > 0)))
        self.final = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 1, 28, 28)
        x = self.conv(x)
        x = self.flatten(x)
        x = self.reduction(x)
        x = self.fraud_layers(x)
        logits = self.final(x)
        return torch.sigmoid(logits)

__all__ = ["FraudLayerParameters", "FraudDetectionHybridModel"]
