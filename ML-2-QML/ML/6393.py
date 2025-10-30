"""Hybrid classical‑quantum classifier with RBF kernel.

This module implements the classical side of the hybrid model. It provides
- a feed‑forward network that mirrors the structure of the quantum ansatz,
- a radial‑basis‑function kernel for similarity learning,
- a build_classifier_circuit helper that returns the network and metadata
  compatible with the quantum helper interface.

The class is fully compatible with the original anchor and can be used
directly as a drop‑in replacement.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn

__all__ = ["HybridClassifierKernel", "build_classifier_circuit"]


class HybridClassifierKernel(nn.Module):
    """Classic feed‑forward network with optional RBF kernel support."""

    def __init__(self, num_features: int, depth: int = 2, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma
        layers = []
        in_dim = num_features
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, num_features))
            layers.append(nn.ReLU())
            in_dim = num_features
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_dim, 2)

    def rbf_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the RBF kernel matrix between two batches."""
        diff = x.unsqueeze(1) - y.unsqueeze(0)
        dist_sq = torch.sum(diff * diff, dim=-1)
        return torch.exp(-self.gamma * dist_sq)

    def forward(self, x: torch.Tensor, support_vectors: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass with optional kernel augmentation."""
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        if support_vectors is not None:
            k = self.rbf_kernel(x, support_vectors)
            logits = torch.cat([logits, k], dim=1)
        return logits


def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
    """Return the classical network and metadata for compatibility."""
    model = HybridClassifierKernel(num_features, depth)
    encoding = list(range(num_features))
    weight_sizes = [m.weight.numel() + m.bias.numel() for m in model.modules() if isinstance(m, nn.Linear)]
    observables = [0, 1]  # placeholder
    return model, encoding, weight_sizes, observables
