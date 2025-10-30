"""
Hybrid transformer that unifies classical and quantum blocks while
remaining fully classical for downstream tasks.  The module is
designed to be drop‑in compatible with the original API, but it
adds two key extensions:
1. A learnable `QuantumEmbedding` that maps token indices to a
   quantum state vector prior to attention.
2. A `QuantumRegularizer` that injects a penalty proportional to the
   circuit depth (or number of parameters) into the loss, allowing
   a fair comparison between classical and quantum sub‑modules.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumEmbedding(nn.Module):
    """
    Lightweight classical wrapper that mimics a quantum embedding.
    It maps the token embedding to a higher dimensional space that
    would be the result of a quantum circuit.
    """
    def __init__(self, embed_dim: int, qubits: int = 2):
        super().__init__()
        if embed_dim % qubits!= 0:
            raise ValueError("embed_dim must be divisible by qubits")
        self.qubits = qubits
        self.n_out = embed_dim // qubits
        self.linear = nn.Linear(embed_dim, self.n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, embed_dim)
        return self.linear(x)


class QuantumRegularizer(nn.Module):
    """
    Computes a simple regularization penalty based on the number of
    trainable parameters in a quantum module.  The penalty can be
    weighted by ``lambda_``.
    """
    def __init__(self, lambda_: float = 1e-4):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, module: nn.Module) -> torch.Tensor:
        n_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return self.lambda_ * torch.tensor(
            n_params, dtype=torch.float32, device=next(module.parameters()).device
        )


class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class MultiHeadAttentionClassical(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_output, _ = self.attn(x, x, x, key_padding_mask=mask)
        return self.dropout(attn_output)


class FeedForwardClassical(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlockClassical(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        return self.norm2(x + ffn_out)


class UnifiedQTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        use_quantum_embedding: bool = False,
        qubits: int = 2,
        lambda_qreg: float = 1e-4,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.use_quantum_embedding = use_quantum_embedding
        if use_quantum_embedding:
            self.quantum_embedding = QuantumEmbedding(embed_dim, qubits)
            self.q_regularizer = QuantumRegularizer(lambda_qreg)
        else:
            self.quantum_embedding = None
            self.q_regularizer = None
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
                for _ in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(
            embed_dim, num_classes if num_classes > 2 else 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq)
        token_emb = self.token_embedding(x)
        if self.use_quantum_embedding:
            token_emb = self.quantum_embedding(token_emb)
        x = self.pos_encoder(token_emb)
        for block in self.transformer_blocks:
            x = block(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)

    def regularization(self) -> torch.Tensor:
        if self.q_regularizer and self.quantum_embedding:
            return self.q_regularizer(self.quantum_embedding)
        return torch.tensor(0.0, device=self.token_embedding.weight.device)


__all__ = [
    "QuantumEmbedding",
    "QuantumRegularizer",
    "PositionalEncoder",
    "MultiHeadAttentionClassical",
    "FeedForwardClassical",
    "TransformerBlockClassical",
    "UnifiedQTransformer",
]
