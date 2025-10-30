"""Hybrid classical classifier with self‑attention preprocessing.

This module defines a single class ``HybridClassifier`` that first applies
a learnable self‑attention block to the raw inputs and then feeds the
result into a configurable feed‑forward network.  The attention block
mirrors the quantum interface but operates purely with PyTorch
tensors, making it a drop‑in replacement for the quantum workflow.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassicalSelfAttention(nn.Module):
    """Learnable self‑attention that mirrors the quantum interface."""

    def __init__(self, embed_dim: int, num_heads: int = 1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, embed_dim)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scores = F.softmax(
            torch.matmul(q, k.transpose(-2, -1)) / (self.embed_dim ** 0.5), dim=-1
        )
        return torch.matmul(scores, v)


class HybridClassifier(nn.Module):
    """Classifier that couples self‑attention with a multilayer MLP."""

    def __init__(self, num_features: int, depth: int, embed_dim: int = 4):
        super().__init__()
        self.attention = ClassicalSelfAttention(embed_dim)
        # Build a configurable feed‑forward network
        layers: list[nn.Module] = []
        in_dim = embed_dim
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, num_features))
            layers.append(nn.ReLU())
            in_dim = num_features
        layers.append(nn.Linear(in_dim, 2))  # binary classification
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            Raw input tensor of shape (batch, embed_dim).

        Returns
        -------
        logits
            Classification logits of shape (batch, 2).
        """
        x = self.attention(x)
        return self.classifier(x)


__all__ = ["HybridClassifier"]
