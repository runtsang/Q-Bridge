"""Enhanced classical convolutional filter with fraud-detection inspired fully connected layer.

The ConvHybrid module fuses a 2‑D convolution with a custom linear layer that mimics the
parameterisation used in the photonic fraud‑detection example.  It can be dropped in
place of the original Conv filter while providing richer expressivity and a clear
interface for quantum‑classical hybrid experiments.
"""

from __future__ import annotations

import torch
from torch import nn
from dataclasses import dataclass
from typing import Tuple


@dataclass
class FraudLayerParameters:
    """Parameters describing a fully connected layer in the classical model."""
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


class FraudLayer(nn.Module):
    """Linear layer with fraud‑detection style weight construction and optional clipping."""

    def __init__(self, params: FraudLayerParameters, *, clip: bool = True) -> None:
        super().__init__()
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
        self.linear = nn.Linear(2, 2)
        with torch.no_grad():
            self.linear.weight.copy_(weight)
            self.linear.bias.copy_(bias)
        self.activation = nn.Tanh()
        self.register_buffer("scale", torch.tensor(params.displacement_r, dtype=torch.float32))
        self.register_buffer("shift", torch.tensor(params.displacement_phi, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out = self.activation(self.linear(x))
        return out * self.scale + self.shift


class ConvHybrid(nn.Module):
    """Convolutional filter followed by a fraud‑detection inspired fully connected block.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the convolutional kernel.
    threshold : float, default 0.0
        Bias applied before the sigmoid activation.
    fraud_params : FraudLayerParameters, optional
        Parameters for the fraud‑detection style layer.  If omitted, defaults to
        a neutral configuration.
    fraud_clip : bool, default True
        Whether to clip the fraud‑layer weights and biases.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        fraud_params: FraudLayerParameters | None = None,
        fraud_clip: bool = True,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        # Convolution produces two feature maps so we can feed them into the
        # fraud‑layer which expects a 2‑dimensional input.
        self.conv = nn.Conv2d(1, 2, kernel_size=kernel_size, bias=True)
        if fraud_params is None:
            fraud_params = FraudLayerParameters(
                bs_theta=0.0,
                bs_phi=0.0,
                phases=(0.0, 0.0),
                squeeze_r=(0.0, 0.0),
                squeeze_phi=(0.0, 0.0),
                displacement_r=(1.0, 1.0),
                displacement_phi=(0.0, 0.0),
                kerr=(0.0, 0.0),
            )
        self.fraud_layer = FraudLayer(fraud_params, clip=fraud_clip)
        self.final_linear = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the hybrid filter."""
        conv_out = self.conv(x)  # shape: (batch, 2, 1, 1)
        conv_out = conv_out.view(conv_out.size(0), 2)
        conv_out = torch.sigmoid(conv_out - self.threshold)
        fraud_out = self.fraud_layer(conv_out)
        return self.final_linear(fraud_out)

    def run(self, data: torch.Tensor) -> float:
        """Convenience wrapper that mimics the original Conv.run API."""
        activations = self.forward(data.unsqueeze(0))
        return activations.mean().item()


__all__ = ["ConvHybrid"]
