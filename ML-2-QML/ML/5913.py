"""Hybrid classical model that interleaves convolution, attention, and quantum‑inspired projections.

The model is intentionally lightweight: a 2‑D CNN extracts spatial features, a self‑attention block re‑weights them, and a final linear head produces a 4‑dimensional embedding.  The attention module is built from the self‑attention helper in the reference pair, but adapted to work with PyTorch tensors and to fit seamlessly into the forward pass.

The design follows the *combination* scaling paradigm: the classical encoder and attention are trained end‑to‑end, while the quantum part is only used for comparative experiments (see the QML counterpart below).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# -- Self‑attention helper ----------------------------------------------------
class ClassicalSelfAttention(nn.Module):
    """Re‑implements the reference self‑attention but as a PyTorch module."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        # Parameters are learnable linear maps that emulate rotation/entangle params.
        self.rotation = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.entangle = nn.Linear(embed_dim, embed_dim * 3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, embed_dim) feature matrix

        Returns:
            attended: (batch, embed_dim) re‑weighted features
        """
        # Compute query, key, value
        q = self.rotation(x)  # (B, 3*D)
        k = self.entangle(x)  # (B, 3*D)
        v = x

        # Reshape to (B, D, 3) for simple dot‑product attention
        q = q.view(-1, self.embed_dim, 3)
        k = k.view(-1, self.embed_dim, 3)

        # Scale and softmax
        scores = torch.softmax(torch.bmm(q, k.transpose(1, 2)) / np.sqrt(self.embed_dim), dim=-1)
        return torch.bmm(scores, v.unsqueeze(-1)).squeeze(-1)


# -- Main hybrid model --------------------------------------------------------
class HybridQuantumNatAttention(nn.Module):
    """Classical CNN + FC + self‑attention backbone."""
    def __init__(self, input_channels: int = 1, embed_dim: int = 64, num_classes: int = 4):
        super().__init__()
        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Global pooling to match self‑attention input size
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # Attention module
        self.attn = ClassicalSelfAttention(embed_dim=16)
        # Projection head
        self.head = nn.Sequential(
            nn.Linear(16, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, num_classes),
        )
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
          1. Convolutional feature extraction.
          2. Global average pooling to (B, 16).
          3. Self‑attention re‑weighting.
          4. Linear head producing 4‑dimensional output.
        """
        bsz = x.size(0)
        feat = self.features(x)          # (B, 16, H', W')
        pooled = self.pool(feat).view(bsz, -1)  # (B, 16)
        attn_out = self.attn(pooled)     # (B, 16)
        logits = self.head(attn_out)     # (B, 4)
        return self.norm(logits)


__all__ = ["HybridQuantumNatAttention"]
