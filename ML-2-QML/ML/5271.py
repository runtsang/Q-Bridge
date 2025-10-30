"""Hybrid fraud detection model combining classical neural network, quantum kernel, and convolutional feature extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, List

import torch
from torch import nn
import numpy as np

# --------------------------------------------------------------------------- #
#   Photonic‑inspired feature extractor
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParameters:
    """Container for a single photonic layer's parameters."""
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


def _build_layer(params: FraudLayerParameters, clip: bool) -> nn.Module:
    weight = torch.tensor(
        [[params.bs_theta, params.bs_phi],
         [params.squeeze_r[0], params.squeeze_r[1]]],
        dtype=torch.float32
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

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = self.activation(self.linear(x))
            return y * self.scale + self.shift

    return Layer()


# --------------------------------------------------------------------------- #
#   Classical convolutional filter (quantum filter emulation)
# --------------------------------------------------------------------------- #
class ConvFilter(nn.Module):
    """Classical emulation of a 2×2 quanvolution filter."""

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def run(self, data) -> float:
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()


# --------------------------------------------------------------------------- #
#   FraudDetectionModel
# --------------------------------------------------------------------------- #
class FraudDetectionModel(nn.Module):
    """Hybrid model that stacks a photonic‑inspired feature extractor,
    an RBF kernel, and a classical convolution filter."""
    def __init__(
        self,
        base_params: FraudLayerParameters,
        hidden_params: Sequence[FraudLayerParameters],
        kernel_gamma: float = 1.0,
        conv_kernel_size: int = 2,
        conv_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        # Photonic feature extractor
        self.feature_extractor = nn.Sequential(
            _build_layer(base_params, clip=False),
            *(_build_layer(p, clip=True) for p in hidden_params),
            nn.Linear(2, 1)
        )
        # RBF kernel mapping
        self.kernel_gamma = kernel_gamma
        # Convolutional filter
        self.conv = ConvFilter(kernel_size=conv_kernel_size, threshold=conv_threshold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: feature extraction → kernel → convolution."""
        feat = self.feature_extractor(x)
        diff = feat - feat.transpose(0, 1)
        k = torch.exp(-self.kernel_gamma * torch.sum(diff * diff, dim=-1, keepdim=True))
        conv_out = self.conv.run(x.detach().cpu().numpy())
        return k.mean() + conv_out

    def predict(self, x: torch.Tensor, shots: int | None = None, seed: int | None = None) -> torch.Tensor:
        """Return predictions with optional shot‑noise simulation."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
        if shots is not None:
            rng = np.random.default_rng(seed)
            noise = rng.normal(0, 1 / shots, size=probs.shape)
            probs = probs + noise
        return probs


__all__ = ["FraudLayerParameters", "FraudDetectionModel", "ConvFilter"]
