"""Hybrid fraud detection model combining classical convolution with a photonic network.

The module defines a `FraudDetectionHybrid` class that first extracts a
convolutional feature using a lightweight PyTorch `ConvFilter` and then
feeds the resulting 2‑component feature vector into a classical
photonic‑inspired network built from `FraudLayerParameters`.  The
construction of each layer is identical to the original `FraudDetection`
seed, but the input is now a learned feature map rather than raw data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

# --------------------------------------------------------------------------- #
# 1.  Parameters and helpers – identical to the original seed
# --------------------------------------------------------------------------- #

@dataclass
class FraudLayerParameters:
    """Parameters describing a fully connected layer in the classical model."""
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
    """Create a single linear‑tanh‑scale layer from parameters."""
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

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Construct a sequential PyTorch model mirroring the layered structure."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# --------------------------------------------------------------------------- #
# 2.  Lightweight convolutional feature extractor
# --------------------------------------------------------------------------- #

class ConvFilter(nn.Module):
    """A minimal 2‑D convolutional filter that mimics the quantum filter."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def run(self, data: torch.Tensor) -> torch.Tensor:
        """
        Process a batch of 2‑D patches.

        Parameters
        ----------
        data : torch.Tensor
            Shape (batch, kernel_size, kernel_size).

        Returns
        -------
        torch.Tensor
            Activation value per sample. Shape (batch,).
        """
        batch = data.shape[0]
        x = data.view(batch, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=(1, 2))

# --------------------------------------------------------------------------- #
# 3.  Hybrid model
# --------------------------------------------------------------------------- #

class FraudDetectionHybrid(nn.Module):
    """
    Hybrid fraud detection model that first extracts a convolutional feature
    and then passes it through a classical photonic‑inspired network.
    """
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layer_params: Iterable[FraudLayerParameters],
        conv_kernel_size: int = 2,
        conv_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv = ConvFilter(kernel_size=conv_kernel_size, threshold=conv_threshold)
        self.model = build_fraud_detection_program(input_params, layer_params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input patches. Shape (batch, 2, 2).

        Returns
        -------
        torch.Tensor
            Model output. Shape (batch, 1).
        """
        conv_out = self.conv.run(x)  # (batch,)
        # Combine with a simple statistic of the raw input
        raw_mean = x.mean(dim=(1, 2))
        features = torch.stack([conv_out, raw_mean], dim=1)
        return self.model(features)

__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "ConvFilter",
    "FraudDetectionHybrid",
]
