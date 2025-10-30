"""
Hybrid classical module that fuses sampling, convolution, and fraud‑detection
layers into one neural network.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Iterable, Tuple, List, Optional


# ----- Convolutional filter -------------------------------------------------
class ConvFilter(nn.Module):
    """A 2‑D convolution that mimics a quantum filter."""

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input of shape (B, 1, H, W) where H=W=kernel_size.

        Returns
        -------
        torch.Tensor
            Mean activation after sigmoid and thresholding, shape (B, 1).
        """
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=(2, 3))


# ----- Fraud‑detection layer -----------------------------------------------
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


class FraudLayer(nn.Module):
    """Implements one fraud‑detection block with linear, activation, and scaling."""

    def __init__(self, params: FraudLayerParameters, clip: bool = True) -> None:
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

        linear = nn.Linear(2, 2)
        with torch.no_grad():
            linear.weight.copy_(weight)
            linear.bias.copy_(bias)

        self.linear = linear
        self.activation = nn.Tanh()
        self.register_buffer("scale", torch.tensor(params.displacement_r))
        self.register_buffer("shift", torch.tensor(params.displacement_phi))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(self.linear(x))
        out = out * self.scale + self.shift
        return out


def build_fraud_detection_module(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Create a sequential fraud‑detection network."""
    modules: List[nn.Module] = [FraudLayer(input_params, clip=False)]
    modules.extend(FraudLayer(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


# ----- Sampler network ------------------------------------------------------
class SamplerNetwork(nn.Module):
    """Simple 2‑class probability sampler."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(x), dim=-1)


# ----- Hybrid module --------------------------------------------------------
class SamplerQNNGen261(nn.Module):
    """
    Composite model that chains a convolution, fraud‑detection,
    and sampler network.  Designed for image‑style inputs of shape
    (B, 1, 2, 2).
    """

    def __init__(
        self,
        conv_kernel_size: int = 2,
        conv_threshold: float = 0.0,
        fraud_input_params: Optional[FraudLayerParameters] = None,
        fraud_layer_params: Optional[List[FraudLayerParameters]] = None,
    ) -> None:
        super().__init__()
        self.conv = ConvFilter(conv_kernel_size, conv_threshold)

        # Default fraud parameters if none provided
        if fraud_input_params is None:
            fraud_input_params = FraudLayerParameters(
                bs_theta=0.5,
                bs_phi=0.5,
                phases=(0.0, 0.0),
                squeeze_r=(0.1, 0.1),
                squeeze_phi=(0.0, 0.0),
                displacement_r=(0.05, 0.05),
                displacement_phi=(0.0, 0.0),
                kerr=(0.0, 0.0),
            )
        if fraud_layer_params is None:
            fraud_layer_params = [
                FraudLayerParameters(
                    bs_theta=0.3,
                    bs_phi=0.3,
                    phases=(0.1, 0.1),
                    squeeze_r=(0.2, 0.2),
                    squeeze_phi=(0.0, 0.0),
                    displacement_r=(0.1, 0.1),
                    displacement_phi=(0.0, 0.0),
                    kerr=(0.0, 0.0),
                )
            ]

        self.fraud = build_fraud_detection_module(fraud_input_params, fraud_layer_params)
        self.sampler = SamplerNetwork()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, 1, H, W) with H=W=2.

        Returns
        -------
        torch.Tensor
            Probabilities of shape (B, 2).
        """
        conv_out = self.conv(x)                # (B, 1)
        fraud_out = self.fraud(conv_out)       # (B, 1)
        # The sampler expects 2‑dimensional inputs; broadcast to match.
        sampler_input = fraud_out.repeat(1, 2)
        return self.sampler(sampler_input)


__all__ = ["SamplerQNNGen261"]
