"""Quantum‑aware self‑attention transformer with a variational circuit
implementation using TorchQuantum.

This module mirrors the classical implementation but replaces the
attention heads and feed‑forward network with quantum‑aware
sub‑modules that can be toggled via the ``use_quantum`` flag.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class MultiHeadAttentionClassical(nn.Module):
    """Standard multi‑head attention implemented with PyTorch."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_output, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_output

class FeedForwardClassical(nn.Module):
    """Two‑layer feed‑forward network."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlockClassical(nn.Module):
    """Standard transformer block with residuals and layer‑norm."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out = self.attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class MultiHeadAttentionQuantum(nn.Module):
    """Quantum‑aware multi‑head attention."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1,
                 q_device: Optional[tq.QuantumDevice] = None):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.q_device = q_device or tq.QuantumDevice(n_wires=self.d_k)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        # Quantum encoder for query
        self.q_encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(self.d_k)]
        )
        self.q_params = nn.Parameter(torch.zeros(self.d_k))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq, _ = x.shape
        k = self.k_linear(x)
        v = self.v_linear(x)
        # Quantum query generation
        q = torch.zeros(batch, seq, self.num_heads, self.d_k, device=x.device)
        for h in range(self.num_heads):
            proj = nn.Linear(self.embed_dim, self.d_k, bias=False)(x)
            q_device = self.q_device.copy(bsz=batch*seq, device=x.device)
            self.q_encoder(q_device, proj.reshape(-1, self.d_k))
            for i, theta in enumerate(self.q_params):
                tq.RX(theta, q_device, wires=i)
            q_h_out = self.measure(q_device)
            q_h_out = q_h_out.reshape(batch, seq, self.d_k)
            q[:, :, h, :] = q_h_out
        k = k.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        q = q.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        out = torch.matmul(scores, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq, self.embed_dim)
        return out

class FeedForwardQuantum(nn.Module):
    """Quantum‑aware feed‑forward network."""
    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        # Quantum encoder for hidden representation
        self.q_encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
        )
        self.q_params = nn.Parameter(torch.zeros(n_qubits))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear1(x)
        out = self.dropout(F.relu(out))
        batch, seq, _ = out.shape
        flat = out.reshape(-1, out.size(-1))
        q_device = tq.QuantumDevice(n_wires=self.q_encoder.n_wires, bsz=batch*seq, device=out.device)
        self.q_encoder(q_device, flat)
        for i, theta in enumerate(self.q_params):
            tq.RX(theta, q_device, wires=i)
        q_out = self.measure(q_device)
        q_out = q_out.reshape(batch, seq, -1)
        out = self.linear2(q_out)
        return out

class TransformerBlockQuantum(nn.Module):
    """Transformer block that uses quantum attention and feed‑forward."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, n_qubits_ffn: int,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out = self.attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class UnifiedSelfAttentionTransformer(nn.Module):
    """
    Hybrid transformer that can switch between classical and quantum blocks.
    """
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 num_blocks: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 n_qubits_ffn: int = 0,
                 use_quantum: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.num_blocks = num_blocks
        self.num_classes = num_classes
        self.dropout = dropout
        self.use_quantum = use_quantum
        self.token_embedding = nn.Embedding(30522, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim)
        self.transformers = nn.ModuleList()
        for _ in range(num_blocks):
            if use_quantum and n_qubits_ffn > 0:
                block = TransformerBlockQuantum(embed_dim, num_heads, ffn_dim, n_qubits_ffn, dropout)
            else:
                block = TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
            self.transformers.append(block)
        self.dropout_layer = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_encoding(tokens)
        for block in self.transformers:
            x = block(x, mask)
        x = x.mean(dim=1)
        x = self.dropout_layer(x)
        return self.classifier(x)

__all__ = [
    "PositionalEncoding",
    "MultiHeadAttentionClassical",
    "FeedForwardClassical",
    "TransformerBlockClassical",
    "MultiHeadAttentionQuantum",
    "FeedForwardQuantum",
    "TransformerBlockQuantum",
    "UnifiedSelfAttentionTransformer",
]
