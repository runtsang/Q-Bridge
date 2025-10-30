from __future__ import annotations

import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable, Sequence

# Classical convolutional filter inspired by the QML Conv reference
class Conv:
    """Return a callable object that emulates the quantum filter with PyTorch ops."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0):
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def run(self, data):
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()

# Photonic‑like layer parameters
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

class FraudDetectionHybrid(nn.Module):
    """
    Hybrid classical model that combines a quantum‑style convolutional filter
    (implemented with a simple Conv2d) and a photonic‑like layered network.
    The Conv filter produces a scalar feature that is concatenated with a
    placeholder value to match the 2‑D input expected by the photonic layers.
    """
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        kernel_size: int = 2,
        threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv = Conv(kernel_size, threshold)
        self.layers = nn.Sequential(
            _layer_from_params(input_params, clip=False),
            *(_layer_from_params(l, clip=True) for l in layers),
            nn.Linear(2, 1),
        )

    def forward(self, data) -> torch.Tensor:
        """
        Args:
            data: 2‑D array with shape (kernel_size, kernel_size) representing
                  a single image patch.

        Returns:
            Tensor of shape (1,) – the fraud‑risk score.
        """
        conv_out = self.conv.run(data)
        # Build a 2‑D input for the photonic layers: [conv_out, 0]
        input_tensor = torch.tensor([conv_out, 0.0], dtype=torch.float32)
        return self.layers(input_tensor)

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    kernel_size: int = 2,
    threshold: float = 0.0,
) -> FraudDetectionHybrid:
    """Convenience wrapper that returns a FraudDetectionHybrid instance."""
    return FraudDetectionHybrid(input_params, layers, kernel_size, threshold)

__all__ = ["FraudLayerParameters", "build_fraud_detection_program", "FraudDetectionHybrid"]
