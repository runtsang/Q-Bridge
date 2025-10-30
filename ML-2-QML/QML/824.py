"""Quantum‑enhanced transformer using Pennylane."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

class MultiHeadAttentionBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        return x.view(batch, -1, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

class QuantumAttentionHead(nn.Module):
    def __init__(self, d_k: int, n_wires: int = 8) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.dev = qml.device("default.qubit", wires=n_wires)
        self.params = nn.Parameter(torch.randn(n_wires, 3))
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")

    def _circuit(self, x: torch.Tensor, params: torch.Tensor) -> list[torch.Tensor]:
        for i in range(self.n_wires):
            qml.RX(x[i], wires=i)
        for i in range(self.n_wires):
            qml.RY(params[i, 0], wires=i)
            qml.RZ(params[i, 1], wires=i)
            qml.RX(params[i, 2], wires=i)
        for i in range(self.n_wires - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[self.n_wires - 1, 0])
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_wires)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = []
        for token in x.unbind(dim=0):
            out.append(self.qnode(token, self.params))
        return torch.stack(out)

class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, n_wires: int = 8) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.combine = nn.Linear(embed_dim, embed_dim, bias=False)
        self.heads = nn.ModuleList([QuantumAttentionHead(self.d_k, n_wires) for _ in range(num_heads)])

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        k, q, v = self.k_linear(x), self.q_linear(x), self.v_linear(x)
        k, q, v = (h.transpose(1, 2) for h in (k, q, v))
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1), -1e9)
        scores = F.softmax(scores, dim=-1)
        weighted_v = torch.matmul(scores, v)
        quantum_out = []
        for head_idx in range(self.num_heads):
            head = weighted_v[:, head_idx, :, :]
            quantum_out.append(self.heads[head_idx](head))
        quantum_out = torch.stack(quantum_out, dim=1)
        quantum_out = quantum_out.transpose(1, 2).contiguous().view(x.shape)
        return self.combine(quantum_out)

class FeedForwardBase(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

class FeedForwardQuantum(FeedForwardBase):
    def __init__(self, embed_dim: int, ffn_dim: int, n_wires: int = 8) -> None:
        super().__init__(embed_dim, ffn_dim)
        self.n_wires = n_wires
        self.dev = qml.device("default.qubit", wires=n_wires)
        self.params = nn.Parameter(torch.randn(n_wires, 3))
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")
        self.linear1 = nn.Linear(n_wires, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def _circuit(self, x: torch.Tensor, params: torch.Tensor) -> list[torch.Tensor]:
        for i in range(self.n_wires):
            qml.RX(x[i], wires=i)
        for i in range(self.n_wires):
            qml.RY(params[i, 0], wires=i)
            qml.RZ(params[i, 1], wires=i)
            qml.RX(params[i, 2], wires=i)
        for i in range(self.n_wires - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[self.n_wires - 1, 0])
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_wires)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = []
        for token in x.unbind(dim=1):
            out.append(self.qnode(token, self.params))
        out = torch.stack(out, dim=1)
        out = self.linear1(out)
        out = self.linear2(F.relu(out))
        return out

class TransformerBlockBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

class TransformerBlockQuantum(TransformerBlockBase):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1, n_wires: int = 8) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout, n_wires)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class TransformerBlockClassical(TransformerBlockBase):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1, depthwise_dropout: float = 0.0) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)
        self.depthwise_dropout = nn.Dropout(depthwise_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.depthwise_dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.depthwise_dropout(ffn_out))

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

class HybridTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        depthwise_dropout: float = 0.0,
        use_quantum: bool = False,
        n_wires: int = 8,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        if use_quantum:
            blocks = [
                TransformerBlockQuantum(
                    embed_dim, num_heads, ffn_dim, dropout, n_wires
                )
                for _ in range(num_blocks)
            ]
        else:
            blocks = [
                TransformerBlockClassical(
                    embed_dim, num_heads, ffn_dim, dropout, depthwise_dropout
                )
                for _ in range(num_blocks)
            ]
        self.transformers = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        x = self.transformers(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)

__all__ = [
    "MultiHeadAttentionBase",
    "MultiHeadAttentionQuantum",
    "FeedForwardBase",
    "FeedForwardQuantum",
    "TransformerBlockBase",
    "TransformerBlockClassical",
    "TransformerBlockQuantum",
    "PositionalEncoder",
    "HybridTransformer",
]
