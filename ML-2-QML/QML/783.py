"""Quantum transformer with variational attention and feed‑forward circuits.

This module implements a fully quantum transformer block that
uses a depth‑controlled variational circuit for each attention head
and a separate variational circuit for the feed‑forward network.
Parameters are shared across heads to keep the model lightweight.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class QAttentionLayer(tq.QuantumModule):
    """Quantum multi‑head attention using a parameter‑shared variational circuit."""
    def __init__(self, d_k: int, num_heads: int, depth: int):
        super().__init__()
        self.d_k = d_k
        self.num_heads = num_heads
        self.depth = depth
        # Encoder maps classical input to quantum state
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(d_k)]
        )
        # Trainable rotations per depth
        self.params = nn.ParameterList(
            [nn.Parameter(torch.randn(d_k)) for _ in range(depth)]
        )

    def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        # x shape: (batch, d_k)
        self.encoder(q_device, x)
        for p in self.params:
            for i in range(self.d_k):
                tq.RX(q_device, wires=i)(p[i])
        # entanglement
        for _ in range(self.depth):
            for i in range(self.d_k - 1):
                tqf.cnot(q_device, wires=[i, i + 1])
        return tq.MeasureAll(tq.PauliZ)(q_device)


class QFeedForwardLayer(tq.QuantumModule):
    """Quantum feed‑forward block with a variational circuit."""
    def __init__(self, n_qubits: int, depth: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.depth = depth
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
        )
        self.params = nn.ParameterList(
            [nn.Parameter(torch.randn(n_qubits)) for _ in range(depth)]
        )

    def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        self.encoder(q_device, x)
        for p in self.params:
            for i in range(self.n_qubits):
                tq.RX(q_device, wires=i)(p[i])
        return tq.MeasureAll(tq.PauliZ)(q_device)


class QuantumTransformerBlock(nn.Module):
    """Transformer block that replaces both attention and feed‑forward with quantum circuits."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 attention_depth: int = 2, ffn_depth: int = 2,
                 n_qubits_ffn: Optional[int] = None,
                 dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        # Quantum attention
        self.attn_layer = QAttentionLayer(self.d_k, num_heads, attention_depth)
        # Quantum feed‑forward
        if n_qubits_ffn is None:
            n_qubits_ffn = self.d_k
        self.ffn_layer = QFeedForwardLayer(n_qubits_ffn, ffn_depth)
        # Classical linear projections to map quantum outputs back to embedding space
        self.proj_out = nn.Linear(self.d_k, embed_dim, bias=False)
        self.ffn_proj = nn.Linear(n_qubits_ffn, ffn_dim, bias=False)
        self.ffn_out_proj = nn.Linear(ffn_dim, embed_dim, bias=False)

    def _apply_q(self, x: torch.Tensor, q_layer: tq.QuantumModule) -> torch.Tensor:
        batch, seq, d = x.shape
        out = []
        for i in range(seq):
            token = x[:, i, :]
            qdev = tq.QuantumDevice(n_wires=d, bsz=batch, device=token.device)
            out.append(q_layer(token, qdev))
        return torch.stack(out, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention
        batch, seq, _ = x.shape
        # split heads
        x_heads = x.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        # apply quantum attention per head
        attn_out_heads = []
        for h in range(self.num_heads):
            head = x_heads[:, h, :, :]
            head_q = self._apply_q(head, self.attn_layer)
            attn_out_heads.append(head_q)
        attn_out = torch.stack(attn_out_heads, dim=1).transpose(1, 2).contiguous()
        attn_out = attn_out.view(batch, seq, self.embed_dim)
        attn_out = self.proj_out(attn_out)
        x = self.norm1(x + self.dropout(attn_out))
        # Feed‑forward
        ffn_out = self._apply_q(x, self.ffn_layer)
        ffn_out = self.ffn_proj(ffn_out)
        ffn_out = F.relu(ffn_out)
        ffn_out = self.ffn_out_proj(ffn_out)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class QuantumTransformer(nn.Module):
    """Stack of QuantumTransformerBlock layers."""
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 attention_depth: int = 2,
                 ffn_depth: int = 2,
                 n_qubits_ffn: Optional[int] = None):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.blocks = nn.ModuleList(
            [QuantumTransformerBlock(embed_dim,
                                     num_heads,
                                     ffn_dim,
                                     attention_depth=attention_depth,
                                     ffn_depth=ffn_depth,
                                     n_qubits_ffn=n_qubits_ffn,
                                     dropout=dropout)
             for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        x = self.pos_encoder(x)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)


__all__ = [
    "QAttentionLayer",
    "QFeedForwardLayer",
    "QuantumTransformerBlock",
    "QuantumTransformer",
    "PositionalEncoder",
]
