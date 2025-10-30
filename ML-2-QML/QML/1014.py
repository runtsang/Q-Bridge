"""Quantum‑classical hybrid transformer with PennyLane integration."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import pennylane.numpy as np


# --------------------------------------------------------------------------- #
# Quantum building blocks (PennyLane)
# --------------------------------------------------------------------------- #
class QAttentionLayer(nn.Module):
    """Quantum circuit that acts as a head in multi‑head attention."""

    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        self.dev = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs: torch.Tensor) -> torch.Tensor:
            for i in range(n_wires):
                qml.RX(inputs[i], wires=i)
            for i in range(n_wires - 1):
                qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(wires=range(n_wires)))

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.circuit(x)


class QFeedForwardLayer(nn.Module):
    """Quantum circuit used in the feed‑forward sub‑module."""

    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        self.dev = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs: torch.Tensor) -> torch.Tensor:
            for i in range(n_wires):
                qml.RY(inputs[i], wires=i)
            for i in range(n_wires - 1):
                qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(wires=range(n_wires)))

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.circuit(x)


# --------------------------------------------------------------------------- #
# Base attention and feed‑forward abstractions
# --------------------------------------------------------------------------- #
class MultiHeadAttentionBase(nn.Module):
    """Base class for all attention variants."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        return x.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)

    def attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        return torch.matmul(scores, v), scores

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Standard multi‑head attention using PyTorch's MultiheadAttention."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_out


class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    """Quantum‑based attention that uses a PennyLane circuit per head."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, n_qubits: int = 8):
        super().__init__(embed_dim, num_heads, dropout)
        self.proj_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.proj_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.proj_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_layers = nn.ModuleList([QAttentionLayer(self.d_k) for _ in range(num_heads)])
        self.combine = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq, _ = x.shape
        q = self.proj_q(x)
        k = self.proj_k(x)
        v = self.proj_v(x)
        q = self.separate_heads(q)
        k = self.separate_heads(k)
        v = self.separate_heads(v)

        outputs = []
        for head_idx in range(self.num_heads):
            head_q = q[:, head_idx]
            head_k = k[:, head_idx]
            head_v = v[:, head_idx]
            # Flatten for quantum processing
            head_q_flat = head_q.reshape(-1, self.d_k)
            head_k_flat = head_k.reshape(-1, self.d_k)
            head_v_flat = head_v.reshape(-1, self.d_k)

            head_q_q = torch.stack([self.q_layers[head_idx](t) for t in head_q_flat])
            head_k_q = torch.stack([self.q_layers[head_idx](t) for t in head_k_flat])
            head_v_q = torch.stack([self.q_layers[head_idx](t) for t in head_v_flat])

            head_q_q = head_q_q.reshape(batch, seq, self.d_k)
            head_k_q = head_k_q.reshape(batch, seq, self.d_k)
            head_v_q = head_v_q.reshape(batch, seq, self.d_k)

            attn_out, _ = self.attention(head_q_q, head_k_q, head_v_q, mask)
            outputs.append(attn_out)

        out = torch.stack(outputs, dim=1)  # (batch, heads, seq, d_k)
        out = out.transpose(1, 2).contiguous().view(batch, seq, self.embed_dim)
        return self.combine(out)


class FeedForwardBase(nn.Module):
    """Base class for feed‑forward layers."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class FeedForwardClassical(FeedForwardBase):
    """Two‑layer perceptron feed‑forward network."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class FeedForwardQuantum(FeedForwardBase):
    """Feed‑forward network realized by a quantum module."""

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.q_layer = QFeedForwardLayer(n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        outputs = []
        for token in x.unbind(dim=1):
            out = self.q_layer(token)
            outputs.append(out)
        out = torch.stack(outputs, dim=1)  # (batch, seq, n_qubits)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))


# --------------------------------------------------------------------------- #
# Transformer block abstractions
# --------------------------------------------------------------------------- #
class TransformerBlockBase(nn.Module):
    """Base transformer block containing attention and feed‑forward parts."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
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
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        n_qubits_transformer: int,
        n_qubits_ffn: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionQuantum(
            embed_dim, num_heads, dropout, n_qubits=n_qubits_transformer
        )
        if n_qubits_ffn > 0:
            self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, dropout)
        else:
            self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


# --------------------------------------------------------------------------- #
# Positional encoding and final classifier
# --------------------------------------------------------------------------- #
class PositionalEncoder(nn.Module):
    """Learnable positional embedding."""

    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(x.size(0), seq_len)
        return x + self.pos_embedding(positions)


class QuantumEnhancedTransformer(nn.Module):
    """Transformer‑based text classifier supporting quantum submodules."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        n_qubits_transformer: int = 0,
        n_qubits_ffn: int = 0,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        if n_qubits_transformer > 0:
            blocks = [
                TransformerBlockQuantum(
                    embed_dim,
                    num_heads,
                    ffn_dim,
                    n_qubits_transformer,
                    n_qubits_ffn,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        else:
            blocks = [
                TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
                for _ in range(num_blocks)
            ]
        self.transformers = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        x = self.transformers(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)


__all__ = [
    "MultiHeadAttentionBase",
    "MultiHeadAttentionClassical",
    "MultiHeadAttentionQuantum",
    "FeedForwardBase",
    "FeedForwardClassical",
    "FeedForwardQuantum",
    "TransformerBlockBase",
    "TransformerBlockClassical",
    "TransformerBlockQuantum",
    "PositionalEncoder",
    "QuantumEnhancedTransformer",
]
