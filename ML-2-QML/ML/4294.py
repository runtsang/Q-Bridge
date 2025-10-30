from __future__ import annotations

import torch
from torch import nn
from dataclasses import dataclass
from typing import List

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

class HybridConvRegressor(nn.Module):
    """
    Classical hybrid model that combines a 2‑D convolutional feature extractor,
    a fraud‑detection style linear block, and a regression head.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        input_params: FraudLayerParameters | None = None,
        hidden_params: List[FraudLayerParameters] | None = None,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        conv_out_dim = kernel_size * kernel_size
        self.reshaper = nn.Linear(conv_out_dim, 2)
        self.input_layer = (
            _layer_from_params(input_params, clip=False) if input_params else nn.Identity()
        )
        self.hidden_layers = (
            nn.Sequential(*(_layer_from_params(p, clip=True) for p in hidden_params))
            if hidden_params
            else nn.Identity()
        )
        self.head = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape (batch, 1, H, W)
        """
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        activations = activations.view(activations.size(0), -1)
        features = self.reshaper(activations)
        out = self.input_layer(features)
        out = self.hidden_layers(out)
        out = self.head(out)
        return out.squeeze(-1)

__all__ = ["HybridConvRegressor", "FraudLayerParameters", "_layer_from_params"]
