"""Hybrid quantum-classical classifier combining self‑attention and a feed‑forward backbone.

The class exposes a `forward` method compatible with PyTorch, and a helper
`build_classifier_circuit` that returns the underlying network and metadata,
mirroring the original anchor interface.  The implementation fuses a
classical self‑attention block (adapted from the SelfAttention seed) with a
layered feed‑forward classifier (adapted from the original ML seed)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["HybridQuantumClassifier", "build_classifier_circuit"]

class ClassicalSelfAttention(nn.Module):
    """Simple self‑attention block used as a preprocessing layer."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, embed_dim]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.embed_dim)
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, v)

class HybridQuantumClassifier(nn.Module):
    """Hybrid classifier that first applies classical self‑attention
    and then a feed‑forward network."""
    def __init__(self, num_features: int, depth: int, embed_dim: int = 4):
        super().__init__()
        self.attention = ClassicalSelfAttention(embed_dim)
        # Feed‑forward backbone
        layers = []
        in_dim = num_features
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, num_features))
            layers.append(nn.ReLU())
            in_dim = num_features
        layers.append(nn.Linear(in_dim, 2))
        self.backbone = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Assume input shape [batch, seq_len, embed_dim] or [batch, num_features]
        if x.dim() == 2:
            x = x.unsqueeze(1)  # treat each sample as seq_len=1
        attn_out = self.attention(x)
        flat = attn_out.view(attn_out.size(0), -1)
        return self.backbone(flat)

def build_classifier_circuit(num_features: int, depth: int) -> tuple:
    """Return a tuple (model, encoding, weight_sizes, observables) compatible
    with the original anchor API."""
    model = HybridQuantumClassifier(num_features, depth)
    # For compatibility we expose dummy metadata
    encoding = list(range(num_features))
    weight_sizes = [p.numel() for p in model.parameters()]
    observables = list(range(2))
    return model, encoding, weight_sizes, observables
