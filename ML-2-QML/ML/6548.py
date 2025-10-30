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

def _classical_layer_from_params(
    params: FraudLayerParameters,
    *,
    clip: bool,
    in_features: int,
    out_features: int,
) -> nn.Module:
    weight = torch.randn(out_features, in_features)
    bias = torch.randn(out_features)
    if clip:
        weight = weight.clamp(-5.0, 5.0)
        bias = bias.clamp(-5.0, 5.0)
    linear = nn.Linear(in_features, out_features)
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
    in_features: int = 2,
    out_features: int = 2,
) -> nn.Sequential:
    modules = [
        _classical_layer_from_params(
            input_params, clip=False, in_features=in_features, out_features=out_features
        )
    ]
    modules.extend(
        _classical_layer_from_params(
            layer, clip=True, in_features=in_features, out_features=out_features
        )
        for layer in layers
    )
    modules.append(nn.Linear(in_features, 1))
    return nn.Sequential(*modules)

class QuanvolutionHybrid(nn.Module):
    """
    Classical hybrid model that first applies a 2×2 convolution to produce
    4 feature maps, optionally passes them through a fraud‑style
    parameterised linear sequence, and finally classifies via a linear head.
    """

    def __init__(
        self,
        num_classes: int = 10,
        fraud_params: FraudLayerParameters | None = None,
        fraud_layers_params: Iterable[FraudLayerParameters] | None = None,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        self.linear = nn.Linear(4 * 14 * 14, num_classes)
        self.fraud_module = None
        if fraud_params is not None:
            self.fraud_module = build_fraud_detection_program(
                fraud_params,
                fraud_layers_params or [],
                in_features=4 * 14 * 14,
                out_features=4 * 14 * 14,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.conv(x)
        features = features.view(x.size(0), -1)
        if self.fraud_module is not None:
            features = self.fraud_module(features)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionHybrid", "FraudLayerParameters", "build_fraud_detection_program"]
