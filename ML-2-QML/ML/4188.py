"""Hybrid classical module exposing self‑attention, RBF kernel and a feed‑forward classifier."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import Iterable, Tuple, Sequence

# --------------------------------------------------------------------------- #
# Classical self‑attention
# --------------------------------------------------------------------------- #
class ClassicalSelfAttention:
    """Standard scaled dot‑product attention."""

    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        query = torch.as_tensor(
            inputs @ rotation_params.reshape(self.embed_dim, -1),
            dtype=torch.float32,
        )
        key = torch.as_tensor(
            inputs @ entangle_params.reshape(self.embed_dim, -1),
            dtype=torch.float32,
        )
        value = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()

# --------------------------------------------------------------------------- #
# RBF kernel
# --------------------------------------------------------------------------- #
class RBFKernel(nn.Module):
    """Radial‑basis kernel used for kernel‑method experiments."""

    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

def kernel_matrix(
    a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0
) -> np.ndarray:
    """Gram matrix between two datasets."""
    kernel = RBFKernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# --------------------------------------------------------------------------- #
# Feed‑forward classifier
# --------------------------------------------------------------------------- #
class FeedForwardClassifier(nn.Module):
    """Simple depth‑controlled classifier."""

    def __init__(self, num_features: int, depth: int):
        super().__init__()
        layers = []
        in_dim = num_features
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, num_features))
            layers.append(nn.ReLU())
            in_dim = num_features
        layers.append(nn.Linear(in_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def build_classifier_circuit(
    num_features: int, depth: int
) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
    """Return a pure‑PyTorch classifier and dummy metadata to keep API parity."""
    network = FeedForwardClassifier(num_features, depth)
    encoding = list(range(num_features))
    weight_sizes = [p.numel() for p in network.parameters()]
    observables = list(range(2))
    return network, encoding, weight_sizes, observables

# --------------------------------------------------------------------------- #
# Hybrid orchestrator
# --------------------------------------------------------------------------- #
class SelfAttentionHybrid:
    """
    Unified classical engine that exposes attention, kernel and classification.
    """

    def __init__(
        self,
        embed_dim: int = 4,
        num_features: int = 4,
        depth: int = 2,
        gamma: float = 1.0,
    ):
        self.attention = ClassicalSelfAttention(embed_dim)
        self.kernel = RBFKernel(gamma)
        self.classifier, self.enc, self.wts, self.obs = build_classifier_circuit(
            num_features, depth
        )

    def run_attention(
        self, inputs: np.ndarray, rotation_params: np.ndarray, entangle_params: np.ndarray
    ) -> np.ndarray:
        """Compute classical self‑attention output."""
        return self.attention.run(rotation_params, entangle_params, inputs)

    def compute_kernel(
        self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]
    ) -> np.ndarray:
        """Return Gram matrix using RBF kernel."""
        return kernel_matrix(a, b, self.kernel.gamma)

    def predict(self, x: np.ndarray) -> torch.Tensor:
        """Feed‑forward classification."""
        x_t = torch.as_tensor(x, dtype=torch.float32)
        return self.classifier(x_t)

__all__ = ["SelfAttentionHybrid"]
