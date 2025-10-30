"""Hybrid transformer with optional quantum attention.

This module extends the original QTransformerTorch API by adding a
`use_quantum` flag.  When `use_quantum=True` each attention head is
implemented as a lightweight Pennylane variational circuit; otherwise
the standard torch.nn.MultiheadAttention is used.  The rest of the
pipeline (position encoding, feed‑forward, classification head)
remains unchanged, enabling effortless comparison between a classical
baseline and a quantum‑enhanced model.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import pennylane.numpy as pnp

# Quantum attention head
class QuantumAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, q_layers: int = 2):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.q_layers = q_layers

        # Parameters for variational layers per head
        self.params = nn.Parameter(torch.randn(num_heads, q_layers, self.head_dim))
        self.dev = qml.device("default.qubit", wires=self.head_dim)

    def _qnode(self, head_idx: int):
        params = self.params[head_idx]

        @qml.qnode(self.dev, interface="torch")
        def circuit(x):
            for i, angle in enumerate(x):
                qml.RX(angle, wires=i)
            for layer in range(self.q_layers):
                for wire in range(self.head_dim):
                    qml.RY(params[layer, wire], wires=wire)
                for wire in range(self.head_dim - 1):
                    qml.CNOT(wires=[wire, wire + 1])
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.head_dim)]

        return circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        # Separate heads
        x_ = x.view(batch, seq, self.num_heads, self.head_dim).transpose(2, 1)
        out_heads = []
        for h in range(self.num_heads):
            tokens = x_[:, h, :, :]  # (batch, seq, head_dim)
            qnode = self._qnode(h)
            flat = tokens.reshape(-1, self.head_dim)
            out_flat = qnode(flat)
            out = out_flat.reshape(batch, seq, self.head_dim)
            out_heads.append(out)
        out_all = torch.stack(out_heads, dim=2).transpose(2, 1).contiguous()
        out_all = out_all.view(batch, seq, self.embed_dim)
        return self.dropout(out_all)

# Hybrid transformer block
class HybridTransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 dropout: float = 0.1, use_quantum: bool = False):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        if use_quantum:
            self.attn = QuantumAttention(embed_dim, num_heads, dropout)
        else:
            self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if isinstance(self.attn, nn.MultiheadAttention):
            attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)
        else:
            attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

# Positional encoding
class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]

# Text classifier
class HybridTextClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        use_quantum: bool = False,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        self.transformers = nn.Sequential(
            *[HybridTransformerBlock(embed_dim, num_heads, ffn_dim,
                                     dropout=dropout,
                                     use_quantum=use_quantum)
              for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        x = self.transformers(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)
