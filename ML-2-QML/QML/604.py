# Quantum transformer implemented with PennyLane.
# Author: OpenAI GPT-oss-20b

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml


# --------------------------------------------------------------------------- #
#  Core building blocks
# --------------------------------------------------------------------------- #
class MultiHeadAttentionBase(nn.Module):
    """Base for all attention layers – the API matches the seed."""
    def __init__(self, embed_dim: int, num_heads: int,
                 dropout: float = 0.1, use_bias: bool = False) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.attn_weights: Optional[torch.Tensor] = None
        self.use_bias = use_bias

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        return x.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)

    def attention(self, query: torch.Tensor, key: torch.Tensor,
                  value: torch.Tensor, mask: Optional[torch.Tensor] = None
                 ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        return torch.matmul(scores, value), scores

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError


class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Standard attention implemented with nn.MultiheadAttention."""
    def __init__(self, embed_dim: int,
                 num_heads: int,
                 dropout: float,
                 use_bias: bool = False) -> None:
        super().__init__(embed_dim, num_heads, dropout, use_bias)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.combine_heads = nn.Linear(embed_dim, embed_dim, bias=use_bias)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        q = self.separate_heads(q)
        k = self.separate_heads(k)
        v = self.separate_heads(v)
        attn_output, _ = self.attention(q, k, v, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(x.size(0), x.size(1), self.embed_dim)
        return self.combine_heads(attn_output)


class QuantumEncoder(nn.Module):
    """Quantum block that maps an input vector to an output vector of the same size."""
    def __init__(self, input_dim: int, output_dim: int, n_wires: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_wires = n_wires
        self.dev = qml.device("default.qubit", wires=n_wires)
        self.qnode = qml.QNode(self.circuit, self.dev, interface="torch")

    def circuit(self, x: torch.Tensor):
        for i in range(min(self.input_dim, self.n_wires)):
            qml.RX(x[i], wires=i)
        for i in range(self.n_wires - 1):
            qml.CNOT(wires=[i, i + 1])
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_wires)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.qnode(x)


class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    """Multi‑head attention that maps projections through a quantum encoder."""
    def __init__(self, embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.1,
                 use_bias: bool = False,
                 n_wires: Optional[int] = None) -> None:
        super().__init__(embed_dim, num_heads, dropout, use_bias)
        if n_wires is None:
            n_wires = embed_dim
        self.q_encoder = QuantumEncoder(embed_dim, embed_dim, n_wires)
        self.k_encoder = QuantumEncoder(embed_dim, embed_dim, n_wires)
        self.v_encoder = QuantumEncoder(embed_dim, embed_dim, n_wires)
        self.combine_heads = nn.Linear(embed_dim, embed_dim, bias=use_bias)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq, _ = x.shape
        def encode_tensor(tensor: torch.Tensor) -> torch.Tensor:
            return self.q_encoder(tensor)
        q = torch.stack([encode_tensor(x[b, s]) for b in range(batch) for s in range(seq)])
        q = q.view(batch, seq, self.embed_dim)
        k = torch.stack([encode_tensor(x[b, s]) for b in range(batch) for s in range(seq)])
        k = k.view(batch, seq, self.embed_dim)
        v = torch.stack([encode_tensor(x[b, s]) for b in range(batch) for s in range(seq)])
        v = v.view(batch, seq, self.embed_dim)
        q = self.separate_heads(q)
        k = self.separate_heads(k)
        v = self.separate_heads(v)
        attn_output, _ = self.attention(q, k, v, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq, self.embed_dim)
        return self.combine_heads(attn_output)


class FeedForwardBase(nn.Module):
    """Base for feed‑forward layers."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class FeedForwardClassical(FeedForwardBase):
    """Two‑layer perceptron feed‑forward network."""
    def __init__(self, embed_dim: int,
                 ffn_dim: int,
                 dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class FeedForwardQuantum(FeedForwardBase):
    """Feed‑forward network realised by a quantum module."""
    def __init__(self, embed_dim: int,
                 ffn_dim: int,
                 n_wires: Optional[int] = None,
                 dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        if n_wires is None:
            n_wires = embed_dim
        self.first_encoder = QuantumEncoder(embed_dim, ffn_dim, n_wires)
        self.second_encoder = QuantumEncoder(ffn_dim, embed_dim, n_wires)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        def encode(tensor: torch.Tensor) -> torch.Tensor:
            return self.first_encoder(tensor)
        out = torch.stack([encode(x[b, s]) for b in range(batch) for s in range(seq)])
        out = out.view(batch, seq, self.ffn_dim)
        out = self.dropout_layer(out)
        def encode2(tensor: torch.Tensor) -> torch.Tensor:
            return self.second_encoder(tensor)
        out = torch.stack([encode2(out[b, s]) for b in range(batch) for s in range(seq)])
        out = out.view(batch, seq, self.embed_dim)
        return out


class TransformerBlockBase(nn.Module):
    """Base transformer block containing attention and feed‑forward parts."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TransformerBlockClassical(TransformerBlockBase):
    def __init__(self, embed_dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class TransformerBlockQuantum(TransformerBlockBase):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 n_wires: Optional[int] = None,
                 dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout, n_wires=n_wires)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_wires=n_wires, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


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
        return x + self.pe[:, :x.size(1)]


class HybridTransformer(nn.Module):
    """Transformer‑based text classifier supporting optional quantum sub‑modules."""
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 use_quantum_attention: bool = False,
                 use_quantum_ffn: bool = False,
                 n_wires: int = 8) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            if use_quantum_attention or use_quantum_ffn:
                self.blocks.append(
                    TransformerBlockQuantum(
                        embed_dim,
                        num_heads,
                        ffn_dim,
                        n_wires=n_wires,
                        dropout=dropout,
                    )
                )
            else:
                self.blocks.append(
                    TransformerBlockClassical(
                        embed_dim,
                        num_heads,
                        ffn_dim,
                        dropout=dropout,
                    )
                )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)  # shape (batch, seq, embed_dim)
        x = self.pos_embedding(x)
        for block in self.blocks:
            x = block(x)
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
    "HybridTransformer",
]
