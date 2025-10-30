"""Enhanced classical classifier with optional attention and hybrid training support."""

from __future__ import annotations

from typing import Iterable, Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumClassifierModel(nn.Module):
    """
    A PyTorch module that mirrors the quantum helper interface but adds
    - a configurable attention‑based feature extractor (optional)
    - a deep MLP head with a *depth*‑layer
    - an interface for hybrid forward passes with a quantum feature vector
    """

    def __init__(
        self,
        num_features: int,
        depth: int = 2,
        attention: bool = False,
        use_hybrid: bool = False,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.depth = depth
        self.attention = attention
        self.use_hybrid = use_hybrid

        # Feature extractor
        if self.attention:
            # Use a single‑head attention over the feature dimension
            self.attn = nn.MultiheadAttention(
                embed_dim=num_features, num_heads=1, batch_first=True
            )
        else:
            self.attn = None

        # MLP head
        layers: List[nn.Module] = []
        in_dim = num_features
        for _ in range(self.depth):
            layers.append(nn.Linear(in_dim, num_features))
            layers.append(nn.ReLU())
            in_dim = num_features
        layers.append(nn.Linear(in_dim, 2))
        self.head = nn.Sequential(*layers)

        # Optional hybrid linear layer if quantum features are concatenated
        if self.use_hybrid:
            self.hybrid_linear = nn.Linear(num_features + 2, 2)

    def get_encoding(self) -> List[int]:
        """Return the encoding mapping for the input features (identity)."""
        return list(range(self.num_features))

    def get_weight_counts(self) -> List[int]:
        """Return the weight count for each linear layer in the model."""
        counts = []
        for m in self.head:
            if isinstance(m, nn.Linear):
                counts.append(m.weight.numel() + m.bias.numel())
        if self.use_hybrid:
            counts.append(self.hybrid_linear.weight.numel() + self.hybrid_linear.bias.numel())
        return counts

    def forward(self, x: torch.Tensor, quantum_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with optional attention and hybrid quantum features.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, num_features).
        quantum_features : Optional[torch.Tensor]
            Tensor of shape (batch, 2) produced by a quantum circuit.
        """
        if self.attn is not None:
            # MultiheadAttention expects (batch, seq_len, embed_dim)
            x = x.unsqueeze(1)  # seq_len=1
            attn_output, _ = self.attn(x, x, x)
            x = attn_output.squeeze(1)
        out = self.head(x)
        if self.use_hybrid and quantum_features is not None:
            # Concatenate quantum output with classical head output
            combined = torch.cat([out, quantum_features], dim=-1)
            out = self.hybrid_linear(combined)
        return out


__all__ = ["QuantumClassifierModel"]
