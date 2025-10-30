from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml


class MultiHeadAttentionBase(nn.Module):
    """Base class for all attention variants."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        use_bias: bool = False,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.use_bias = use_bias

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        return (
            x.view(batch_size, seq_len, self.num_heads, self.d_k)
           .transpose(1, 2)
           .contiguous()
        )

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, seq_len, d_k = x.size()
        return (
            x.transpose(1, 2)
           .contiguous()
           .view(batch_size, seq_len, self.num_heads * d_k)
        )

    def forward(self, *args, **kwargs) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Standard multi‑head attention implemented in pure PyTorch."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        use_bias: bool = True,
    ) -> None:
        super().__init__(embed_dim, num_heads, dropout, use_bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=use_bias)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = self.separate_heads(q)
        k = self.separate_heads(k)
        v = self.separate_heads(v)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_proj(out)


class QLayer(nn.Module):
    """Quantum layer that maps a d‑dim vector to a new d‑dim vector via a small variational circuit."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.d = d
        self.dev = qml.device("default.qubit", wires=d)
        self.params = nn.Parameter(torch.randn(d))

        @qml.qnode(self.dev, interface="torch")
        def circuit(x, params):
            for i in range(self.d):
                qml.RX(x[i], wires=i)
            for i in range(self.d):
                qml.RY(params[i], wires=i)
            return [qml.expval(qml.PauliZ(i)) for i in range(self.d)]

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, d)
        outputs = []
        for i in range(x.size(0)):
            outputs.append(self.circuit(x[i], self.params))
        return torch.stack(outputs)


class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    """Multi‑head attention where each head projection is processed by a quantum circuit."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        use_bias: bool = False,
    ) -> None:
        super().__init__(embed_dim, num_heads, dropout, use_bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.q_layer = QLayer(self.d_k)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k)

        # Reshape to (batch*seq*heads, d_k) for quantum processing
        q_flat = q.reshape(-1, self.d_k)
        k_flat = k.reshape(-1, self.d_k)
        v_flat = v.reshape(-1, self.d_k)

        q_q = self.q_layer(q_flat)
        k_q = self.q_layer(k_flat)
        v_q = self.q_layer(v_flat)

        q = q_q.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        k = k_q.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        v = v_q.reshape(batch_size, seq_len, self.num_heads, self.d_k)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_proj(out)


class FeedForwardBase(nn.Module):
    """Base class for feed‑forward networks."""

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
    """Feed‑forward network realised by a quantum module."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.q_layer = QLayer(embed_dim)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.size()
        flat = x.view(-1, self.embed_dim)
        qout = self.q_layer(flat)
        hidden = F.relu(self.linear1(qout))
        out = self.linear2(hidden)
        return out.view(batch, seq_len, self.embed_dim)


class TransformerBlockBase(nn.Module):
    """Base transformer block containing attention and feed‑forward parts."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class TransformerBlockClassical(TransformerBlockBase):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ) -> None:
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
        dropout: float = 0.1,
    ) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, dropout)

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
    """Transformer‑based text classifier supporting optional quantum sub‑modules."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        quantum: bool = False,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        if quantum:
            self.transformers = nn.Sequential(
                *[
                    TransformerBlockQuantum(embed_dim, num_heads, ffn_dim, dropout)
                    for _ in range(num_blocks)
                ]
            )
        else:
            self.transformers = nn.Sequential(
                *[
                    TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
                    for _ in range(num_blocks)
                ]
            )
        self.dropout = nn.Dropout(dropout)
        self.classifier = (
            nn.Linear(embed_dim, num_classes)
            if num_classes > 2
            else nn.Linear(embed_dim, 1)
        )

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
    "TextClassifier",
]
