"""Hybrid transformer with quantum kernel support."""

from __future__ import annotations

import math
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import numpy as np


# --------------------------------------------------------------------------- #
# Quantum kernel utilities (inspired by QuantumKernelMethod.py)
# --------------------------------------------------------------------------- #
class QuantumKernelAnsatz(tq.QuantumModule):
    """Encodes two classical vectors via a fixed circuit and returns the overlap."""
    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_wires)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(1)
        self.encoder(q_device, x)
        self.encoder(q_device, y, params=-y)
        self.measure(q_device)


class QuantumKernel(tq.QuantumModule):
    """Quantum kernel that evaluates the overlap between two vectors."""
    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=n_wires)
        self.ansatz = QuantumKernelAnsatz(n_wires)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, self.n_wires)
        y = y.reshape(-1, self.n_wires)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])


def quantum_kernel_matrix(a: Iterable[torch.Tensor], b: Iterable[torch.Tensor], n_wires: int = 4) -> np.ndarray:
    """Compute Gram matrix using the quantum kernel."""
    kernel = QuantumKernel(n_wires)
    return np.array([[kernel(x, y).item() for y in b] for x in a])


# --------------------------------------------------------------------------- #
# Positional encoding (identical to the classical version)
# --------------------------------------------------------------------------- #
class PositionalEncoder(nn.Module):
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


# --------------------------------------------------------------------------- #
# Attention modules
# --------------------------------------------------------------------------- #
class MultiHeadAttentionBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError


class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_output, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_output


class QuantumKernelAttention(MultiHeadAttentionBase):
    """Attention that uses a quantum kernel to compute similarity between queries and keys."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, n_wires: int = 4) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.n_wires = n_wires
        self.q_kernel = QuantumKernel(n_wires)
        self.d_k = embed_dim // num_heads

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq_len, _ = x.size()
        x = x.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # (batch, heads, seq_len, d_k)
        scores = torch.zeros(batch, self.num_heads, seq_len, seq_len, device=x.device)
        for b in range(batch):
            for h in range(self.num_heads):
                for i in range(seq_len):
                    for j in range(seq_len):
                        scores[b, h, i, j] = self.q_kernel(x[b, h, i], x[b, h, j])
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, 0.0)
        scores = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(scores, x)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
        return out


# --------------------------------------------------------------------------- #
# Feed‑forward modules
# --------------------------------------------------------------------------- #
class FeedForwardBase(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class FeedForwardClassical(FeedForwardBase):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class QuantumKernelFeedForward(FeedForwardBase):
    """Feed‑forward that maps input through a quantum kernel matrix before linear layers."""
    def __init__(self, embed_dim: int, ffn_dim: int, n_wires: int = 4, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.n_wires = n_wires
        self.q_kernel = QuantumKernel(n_wires)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.size()
        flat = x.reshape(batch * seq_len, -1)
        n = flat.size(0)
        gram = torch.zeros(n, n, device=x.device, dtype=x.dtype)
        for i in range(n):
            for j in range(n):
                gram[i, j] = self.q_kernel(flat[i], flat[j])
        out = torch.matmul(gram, flat)
        out = out.reshape(batch, seq_len, -1)
        return self.linear2(self.dropout(F.relu(self.linear1(out))))


# --------------------------------------------------------------------------- #
# Transformer blocks
# --------------------------------------------------------------------------- #
class TransformerBlockBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TransformerBlockClassical(TransformerBlockBase):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class TransformerBlockQuantum(TransformerBlockBase):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, n_wires: int = 4, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = QuantumKernelAttention(embed_dim, num_heads, dropout, n_wires)
        self.ffn = QuantumKernelFeedForward(embed_dim, ffn_dim, n_wires, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


# --------------------------------------------------------------------------- #
# Hybrid transformer
# --------------------------------------------------------------------------- #
class HybridTransformer(nn.Module):
    """Quantum‑enhanced transformer with optional classical or kernel sub‑modules."""
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
        n_wires: int = 4,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        blocks = []
        for _ in range(num_blocks):
            if use_quantum:
                blocks.append(TransformerBlockQuantum(embed_dim, num_heads, ffn_dim, n_wires=n_wires, dropout=dropout))
            else:
                blocks.append(TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout=dropout))
        self.transformer = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)


__all__ = [
    "QuantumKernelAnsatz",
    "QuantumKernel",
    "quantum_kernel_matrix",
    "PositionalEncoder",
    "MultiHeadAttentionBase",
    "MultiHeadAttentionClassical",
    "QuantumKernelAttention",
    "FeedForwardBase",
    "FeedForwardClassical",
    "QuantumKernelFeedForward",
    "TransformerBlockBase",
    "TransformerBlockClassical",
    "TransformerBlockQuantum",
    "HybridTransformer",
]
