"""
FraudDetectionHybrid.py

A hybrid classical fraud detection architecture that combines:
  1. Photonic‑style fully‑connected layers (classical analogue).
  2. A self‑attention module for contextual feature weighting.
  3. A lightweight linear classifier.

The design mirrors the original seed but adds a self‑attention block and optional
quanvolution filter for image‑style inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn
import numpy as np


# --------------------------------------------------------------------------- #
# Photonic‑style layer utilities
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParameters:
    """Parameters describing a fully‑connected photonic layer."""
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
            return outputs * self.scale + self.shift

    return Layer()


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Create a sequential PyTorch model mirroring the layered structure."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


# --------------------------------------------------------------------------- #
# Self‑attention module
# --------------------------------------------------------------------------- #
class ClassicalSelfAttention(nn.Module):
    """A lightweight self‑attention block that operates on 2‑dimensional embeddings."""

    def __init__(self, embed_dim: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, embed_dim)
        Returns:
            Tensor of shape (batch, embed_dim) after attention weighting.
        """
        query = self.query_proj(x)
        key = self.key_proj(x)
        value = self.value_proj(x)
        scores = torch.softmax(
            torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.embed_dim), dim=-1
        )
        return torch.matmul(scores, value)


# --------------------------------------------------------------------------- #
# Optional quanvolution filter for image‑style data
# --------------------------------------------------------------------------- #
class QuanvolutionFilter(nn.Module):
    """Classical 2‑D convolution that mimics a quanvolution kernel."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.conv(x)
        return features.view(x.size(0), -1)


# --------------------------------------------------------------------------- #
# Full hybrid fraud detection model
# --------------------------------------------------------------------------- #
class FraudDetectionHybrid(nn.Module):
    """
    End‑to‑end classical fraud detection model that:
      * extracts features via photonic‑style layers
      * refines them with self‑attention
      * optionally applies a quanvolution filter for image inputs
      * classifies with a linear head
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        use_quanvolution: bool = False,
    ) -> None:
        super().__init__()
        self.feature_extractor = build_fraud_detection_program(input_params, layers)
        self.attention = ClassicalSelfAttention(embed_dim=4)
        self.use_quanvolution = use_quanvolution
        if use_quanvolution:
            self.quanvolution = QuanvolutionFilter()
        self.classifier = nn.Linear(4, 2)  # binary output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, 2) for tabular data
               or (batch, 1, 28, 28) for image data when use_quanvolution=True
        Returns:
            Log‑probabilities of shape (batch, 2)
        """
        if self.use_quanvolution:
            # Image path
            features = self.quanvolution(x)
        else:
            # Tabular path
            features = self.feature_extractor(x)
        # Self‑attention refines the embedding
        features = self.attention(features)
        logits = self.classifier(features)
        return torch.nn.functional.log_softmax(logits, dim=-1)


__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "ClassicalSelfAttention",
    "QuanvolutionFilter",
    "FraudDetectionHybrid",
]
