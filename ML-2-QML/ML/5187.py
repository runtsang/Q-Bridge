"""Hybrid binary classifier implemented purely with PyTorch.

This module builds upon transformer encoders and a lightweight
classical head inspired by the quantum expectation layer from the
original hybrid design.  The architecture is deliberately simple
so that it can be swapped for a quantum‑enabled variant without
changing the external API.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple

@dataclass
class FraudLayerParameters:
    """Placeholder for fraud‑detection style parameters.
    Included only for API compatibility; not used in the ML version."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding as in the transformer literature."""
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) *
                             (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class TransformerEncoder(nn.Module):
    """Wrapper around nn.TransformerEncoder that accepts the same
    hyper‑parameters as the quantum version."""
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 num_layers: int,
                 ffn_dim: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        return self.encoder(src)

class HybridHead(nn.Module):
    """Classical head that mimics the quantum expectation output
    with a sigmoid activation."""
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x).squeeze(-1)
        return torch.sigmoid(logits + self.shift)

class HybridBinaryClassifier(nn.Module):
    """End‑to‑end binary classifier that can be instantiated
    in either the classical or quantum module."""
    def __init__(self,
                 input_dim: int,
                 seq_len: int,
                 embed_dim: int,
                 num_heads: int,
                 num_layers: int,
                 ffn_dim: int,
                 shift: float = 0.0) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)
        self.encoder = TransformerEncoder(embed_dim, num_heads, num_layers, ffn_dim)
        self.head = HybridHead(embed_dim, shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, input_dim)
        Returns:
            Tensor of shape (batch, 2) with class probabilities.
        """
        # Project input to embedding space
        x = self.input_proj(x)            # (batch, seq_len, embed_dim)
        x = self.pos_encoder(x)
        # Transformer expects (seq_len, batch, embed_dim) for default
        x = x.transpose(0, 1)              # (seq_len, batch, embed_dim)
        x = self.encoder(x)                # (seq_len, batch, embed_dim)
        x = x.transpose(0, 1).mean(dim=1)  # (batch, embed_dim)
        prob = self.head(x)                # (batch,)
        return torch.stack([prob, 1 - prob], dim=-1)

__all__ = ["HybridBinaryClassifier", "FraudLayerParameters"]
