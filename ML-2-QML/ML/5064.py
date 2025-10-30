"""Hybrid classical convolutional regression with fraud detection."""

from __future__ import annotations

import torch
from torch import nn
from typing import Iterable, Sequence

# Classical convolution filter
class ConvFilter(nn.Module):
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # data shape (batch, 1, H, W)
        logits = self.conv(data)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=[1, 2, 3])  # scalar per sample

# Classical RBF kernel
class RBFKernel(nn.Module):
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

# Classical regression head
class RegressionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int] | tuple[int,...] = (32, 16)) -> None:
        super().__init__()
        layers = []
        last_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        layers.append(nn.Linear(last_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

# Fraud‑detection layers
class FraudLayerParameters:
    def __init__(self, bs_theta: float, bs_phi: float, phases: tuple[float, float],
                 squeeze_r: tuple[float, float], squeeze_phi: tuple[float, float],
                 displacement_r: tuple[float, float], displacement_phi: tuple[float, float],
                 kerr: tuple[float, float]) -> None:
        self.bs_theta = bs_theta
        self.bs_phi = bs_phi
        self.phases = phases
        self.squeeze_r = squeeze_r
        self.squeeze_phi = squeeze_phi
        self.displacement_r = displacement_r
        self.displacement_phi = displacement_phi
        self.kerr = kerr

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    weight = torch.tensor(
        [[params.bs_theta, params.bs_phi],
         [params.squeeze_r[0], params.squeeze_r[1]]], dtype=torch.float32)
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

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# Combined hybrid model
class HybridConvolutionalRegressor(nn.Module):
    def __init__(self,
                 conv_kernel_size: int = 2,
                 conv_threshold: float = 0.0,
                 gamma: float = 1.0,
                 regression_hidden: list[int] | tuple[int,...] = (32, 16),
                 fraud_input: FraudLayerParameters | None = None,
                 fraud_layers: list[FraudLayerParameters] | None = None):
        super().__init__()
        self.conv = ConvFilter(conv_kernel_size, conv_threshold)
        self.kernel = RBFKernel(gamma)
        self.regression = RegressionHead(input_dim=1, hidden_dims=regression_hidden)
        if fraud_input is None:
            fraud_input = FraudLayerParameters(
                bs_theta=0.0, bs_phi=0.0,
                phases=(0.0, 0.0),
                squeeze_r=(0.0, 0.0),
                squeeze_phi=(0.0, 0.0),
                displacement_r=(0.0, 0.0),
                displacement_phi=(0.0, 0.0),
                kerr=(0.0, 0.0))
            fraud_layers = []
        self.fraud_module = build_fraud_detection_program(fraud_input, fraud_layers)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        # images: (batch, 1, H, W)
        conv_output = self.conv(images)  # (batch,)
        # Kernel similarity against itself as reference
        kernel_matrix = self.kernel(conv_output.unsqueeze(1), conv_output.unsqueeze(0))
        features = kernel_matrix.mean(dim=1)  # aggregate similarity
        preds = self.regression(features.unsqueeze(1))
        fraud_score = self.fraud_module(features.unsqueeze(1))
        return {"prediction": preds, "fraud_score": fraud_score.squeeze(-1)}

__all__ = ["HybridConvolutionalRegressor", "FraudLayerParameters", "build_fraud_detection_program"]
