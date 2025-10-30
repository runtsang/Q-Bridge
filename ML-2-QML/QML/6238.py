"""Quantum‑enhanced transformer using TorchQuantum.

The implementation mirrors the classical version but replaces each
attention head with a small 4‑qubit variational circuit and the feed‑forward
sub‑layer with a 6‑qubit circuit. The quantum modules are fully trainable
and can be swapped in or out by adjusting the constructor flags.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class MultiHeadAttentionBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError(
                f"Embedding dimension ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        if self.d_k!= 4:
            raise ValueError("For the quantum head implementation, d_k must be 4.")
        self.dropout = nn.Dropout(dropout)

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

    def attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        return torch.matmul(scores, value), scores


class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    """Attention where each head is a 4‑qubit variational circuit."""

    class _HeadQModule(tq.QuantumModule):
        def __init__(self, n_wires: int = 4) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.params = nn.ModuleList(
                [tq.RY(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for wire, gate in enumerate(self.params):
                gate(q_device, wires=wire)
            return self.measure(q_device)

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.heads = nn.ModuleList([self._HeadQModule() for _ in range(num_heads)])
        self.combine_heads = nn.Linear(embed_dim, embed_dim)

    def _quantum_head(self, x_slice: torch.Tensor) -> torch.Tensor:
        """
        Run a single head’s quantum circuit on a batch of tokens.

        x_slice: Tensor of shape (batch, seq_len, d_k)
        """
        batch, seq_len, d_k = x_slice.shape
        flat = x_slice.reshape(batch * seq_len, d_k)
        qdev = tq.QuantumDevice(
            n_wires=self.heads[0].n_wires, bsz=flat.size(0), device=flat.device
        )
        out = self.heads[0](flat, qdev)
        return out.reshape(batch, seq_len, d_k)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.size()
        if embed_dim!= self.embed_dim:
            raise ValueError(
                f"Input embedding ({embed_dim}) does not match layer embedding size ({self.embed_dim})"
            )
        # Slice the embedding into head‑wise segments
        slices = torch.chunk(x, self.num_heads, dim=2)
        # Apply quantum circuit to each head slice
        projections = [self._quantum_head(s) for s in slices]
        # Stack heads: (batch, seq_len, num_heads, d_k)
        proj = torch.stack(projections, dim=2)
        # Reshape for attention computation
        proj = proj.view(batch_size, seq_len, self.num_heads, self.d_k)
        proj = proj.transpose(1, 2)  # (batch, heads, seq, d_k)
        # Compute attention
        scores = torch.matmul(proj, proj.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, proj)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.combine_heads(out)


class FeedForwardBase(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class FeedForwardQuantum(FeedForwardBase):
    """Feed‑forward block implemented with a 6‑qubit variational circuit."""

    class _QuantumFFN(tq.QuantumModule):
        def __init__(self, n_wires: int = 6) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.params = nn.ModuleList(
                [tq.RY(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for wire, gate in enumerate(self.params):
                gate(q_device, wires=wire)
            return self.measure(q_device)

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int = 6, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        if n_qubits!= ffn_dim:
            raise ValueError("For the quantum feed‑forward, n_qubits must equal ffn_dim (default 6).")
        self.quantum_module = self._QuantumFFN(n_qubits)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        # Project to ffn_dim
        proj = self.linear1(x)
        flat = proj.reshape(batch_size * seq_len, self.ffn_dim)
        qdev = tq.QuantumDevice(
            n_wires=self.quantum_module.n_wires, bsz=flat.size(0), device=flat.device
        )
        out = self.quantum_module(flat, qdev)
        out = out.reshape(batch_size, seq_len, self.ffn_dim)
        out = self.linear2(self.dropout(out))
        return F.relu(out)


class TransformerBlockQuantum(nn.Module):
    """Quantum‑enhanced transformer block."""

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding identical to the classical version."""

    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
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


class TextClassifier(nn.Module):
    """Hybrid transformer classifier that uses the quantum block by default."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.transformer_layers = nn.Sequential(
            *[
                TransformerBlockQuantum(embed_dim, num_heads, ffn_dim, dropout)
                for _ in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_layers(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)


__all__ = [
    "MultiHeadAttentionBase",
    "MultiHeadAttentionQuantum",
    "FeedForwardBase",
    "FeedForwardQuantum",
    "TransformerBlockQuantum",
    "PositionalEncoder",
    "TextClassifier",
]
