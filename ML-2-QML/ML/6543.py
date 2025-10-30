"""Hybrid classical model blending CNN feature extraction and quantum‑inspired transformations.

This module builds on the original Quantum‑NAT CNN and extends it with a
parameter‑free “quantum‑inspired” layer that mimics the effect of a
variational circuit: a random linear map followed by a sinusoidal
non‑linearity.  The design allows the network to learn richer feature
representations while keeping the forward pass fully classical and
GPU‑friendly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import math


class QuantumInspiredLayer(nn.Module):
    """A lightweight classical surrogate for a variational circuit.

    The layer applies a random linear transformation followed by a
    sin‑cos mixture.  The random weights are fixed (non‑trainable) to
    emulate a random circuit, while the sine/cos activation allows
    the network to learn constructive interference patterns.
    """

    def __init__(self, in_features: int, out_features: int = 32) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # fixed random weight matrix
        self.register_buffer(
            "W",
            torch.randn(in_features, out_features, dtype=torch.float32)
            / math.sqrt(in_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_features)
        z = x @ self.W  # (batch, out_features)
        # sinusoidal non‑linearity
        return torch.sin(z) + torch.cos(z)


class HybridNATModel(nn.Module):
    """Hybrid classical model for the Quantum‑NAT benchmark.

    Architecture:
        1. 2‑layer CNN to extract local image features.
        2. Quantum‑inspired layer to inject global, non‑linear interactions.
        3. Linear classifier producing 4 logits.
    """

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        # CNN backbone (identical to the original QFCModel)
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # compute the flattened size after convs: 16 * 7 * 7 for 28x28 input
        self._flattened = 16 * 7 * 7
        # quantum‑inspired layer
        self.qinspired = QuantumInspiredLayer(self._flattened, 64)
        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes),
        )
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        feat = self.features(x)
        flat = feat.view(bsz, -1)
        qi = self.qinspired(flat)
        logits = self.classifier(qi)
        return self.norm(logits)


__all__ = ["HybridNATModel"]
