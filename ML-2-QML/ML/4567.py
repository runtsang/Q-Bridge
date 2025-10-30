"""Hybrid classical fully‑connected attention classifier.

This module brings together the three seed projects:
- A linear embedding (from *FCL.py*).
- A self‑attention block (from *SelfAttention.py*).
- A two‑layer feed‑forward classifier (from *QuantumClassifierModel.py*).

The :class:`HybridFCL` class exposes a single ``run`` method that accepts a
NumPy array of inputs and returns logits.  All sub‑components are wrapped
inside a single PyTorch ``nn.Module`` to keep the interface identical to the
original seeds.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn.functional import softmax

__all__ = ["HybridFCL"]


class HybridFCL(nn.Module):
    """
    Hybrid classical fully connected attention classifier.

    Parameters
    ----------
    n_features : int
        Number of input features.
    embed_dim : int, default 4
        Dimension of the embedding used by the attention block.
    depth : int, default 2
        Depth of the feed‑forward classifier.
    n_classes : int, default 2
        Number of output classes.
    """

    def __init__(
        self,
        n_features: int,
        embed_dim: int = 4,
        depth: int = 2,
        n_classes: int = 2,
    ) -> None:
        super().__init__()
        # 1. Linear embedding
        self.embedding = nn.Linear(n_features, embed_dim)

        # 2. Self‑attention
        self.attn = self._build_attention(embed_dim)

        # 3. Feed‑forward classifier
        self.classifier = self._build_classifier(embed_dim, depth, n_classes)

    @staticmethod
    def _build_attention(embed_dim: int) -> nn.Module:
        """Replicates the SelfAttention logic as a PyTorch module."""
        class SelfAttentionModule(nn.Module):
            def __init__(self, dim: int):
                super().__init__()
                self.q_proj = nn.Linear(dim, dim, bias=False)
                self.k_proj = nn.Linear(dim, dim, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # x shape: (batch, dim)
                q = self.q_proj(x)
                k = self.k_proj(x)
                v = x
                scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(dim)
                attn = softmax(scores, dim=-1)
                return torch.matmul(attn, v)

        return SelfAttentionModule(embed_dim)

    @staticmethod
    def _build_classifier(
        in_dim: int, depth: int, n_classes: int
    ) -> nn.Module:
        """Constructs a shallow feed‑forward network."""
        layers = []
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, in_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(in_dim, n_classes))
        return nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the hybrid model.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (batch, n_features).

        Returns
        -------
        torch.Tensor
            Logits of shape (batch, n_classes).
        """
        x = self.embedding(inputs)
        x = self.attn(x)
        logits = self.classifier(x)
        return logits

    def run(self, inputs: np.ndarray) -> np.ndarray:
        """Convenience wrapper that accepts a NumPy array."""
        self.eval()
        with torch.no_grad():
            inp = torch.as_tensor(inputs, dtype=torch.float32)
            logits = self.forward(inp)
            return logits.numpy()
