"""
HybridSelfAttentionQNN – Classical side
Author: GPT‑OSS‑20B
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridSelfAttentionQNN(nn.Module):
    """
    Classical hybrid model that combines a transformer‑style self‑attention block
    with a QCNN‑inspired fully‑connected encoder.
    The design mirrors the original SelfAttention and QCNN seeds while adding
    multi‑head attention, residual connections and dropout for robustness.
    """

    def __init__(
        self,
        embed_dim: int,
        heads: int = 4,
        dropout: float = 0.1,
        qcnn_arch: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        self.dropout = dropout

        # Self‑attention block
        self.self_attn = nn.MultiheadAttention(
            embed_dim, heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Feed‑forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

        # QCNN‑style encoder (optional)
        if qcnn_arch is None:
            qcnn_arch = [embed_dim, embed_dim // 2, embed_dim // 4]
        self.qcnn = self._build_qcnn_encoder(qcnn_arch)

        self.dropout_layer = nn.Dropout(dropout)

    def _build_qcnn_encoder(self, arch: list[int]) -> nn.Module:
        layers = nn.ModuleList()
        for in_dim, out_dim in zip(arch[:-1], arch[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.Tanh())
        # Simple pooling (down‑sampling)
        layers.append(nn.Linear(arch[-1], arch[-1] // 2))
        layers.append(nn.Tanh())
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, embed_dim)
        """
        # Self‑attention with residual
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(attn_out + x)

        # Feed‑forward with residual
        ffn_out = self.ffn(x)
        x = self.norm2(ffn_out + x)

        # Flatten for QCNN encoder
        batch, seq, dim = x.shape
        x = x.reshape(batch * seq, dim)

        # QCNN encoder
        x = self.qcnn(x)

        # Reshape back
        x = x.reshape(batch, seq, -1)
        return self.dropout_layer(x)

__all__ = ["HybridSelfAttentionQNN"]
