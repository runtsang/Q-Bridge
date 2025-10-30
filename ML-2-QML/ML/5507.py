"""Hybrid classical fraud detection model combining CNN, self‑attention and a feed‑forward classifier.

The module can be trained independently or used as a feature extractor for a quantum layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class FraudLayerParameters:
    """Parameters for a single photonic‑style layer (used only for interface)."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


class ClassicalSelfAttention(nn.Module):
    """Self‑attention module that mirrors the quantum implementation."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, embed_dim]
        query = self.query_proj(x)
        key = self.key_proj(x)
        value = self.value_proj(x)
        scores = F.softmax(query @ key.transpose(-1, -2) / np.sqrt(self.embed_dim), dim=-1)
        return scores @ value


def build_classifier_circuit(num_features: int, depth: int) -> nn.Sequential:
    """Construct a depth‑controlled feed‑forward classifier."""
    layers: List[nn.Module] = []
    in_dim = num_features
    for _ in range(depth):
        layers.append(nn.Linear(in_dim, num_features))
        layers.append(nn.ReLU())
        in_dim = num_features
    layers.append(nn.Linear(in_dim, 2))  # binary fraud decision
    return nn.Sequential(*layers)


class FraudDetectionHybrid(nn.Module):
    """Comprehensive classical fraud detection model."""

    def __init__(self,
                 cnn_channels: int = 8,
                 cnn_depth: int = 2,
                 attention_dim: int = 64,
                 classifier_depth: int = 3):
        super().__init__()
        # Convolutional feature extractor
        conv_layers: List[nn.Module] = [
            nn.Conv2d(1, cnn_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)]
        for _ in range(cnn_depth - 1):
            conv_layers.extend([
                nn.Conv2d(cnn_channels, cnn_channels * 2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)])
        self.features = nn.Sequential(*conv_layers)

        # Self‑attention on flattened features
        self.attention = ClassicalSelfAttention(attention_dim)

        # Classifier
        # Feature dimension after conv layers depends on input size; assume 28x28 input
        flattened_dim = cnn_channels * (28 // (2 ** cnn_depth)) ** 2
        self.classifier = build_classifier_circuit(flattened_dim, classifier_depth)

        self.norm = nn.BatchNorm1d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, 1, H, W]
        feats = self.features(x)
        flat = feats.view(feats.size(0), -1)
        attn_out = self.attention(flat)
        logits = self.classifier(attn_out)
        return self.norm(logits)


__all__ = ["FraudLayerParameters", "ClassicalSelfAttention",
           "build_classifier_circuit", "FraudDetectionHybrid"]
