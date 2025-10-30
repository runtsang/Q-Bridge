"""
QuantumTransformerHybrid: Classical transformer with optional quantum heads.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml


# --------------------------------------------------------------------------- #
#  Base classes
# --------------------------------------------------------------------------- #
class MultiHeadAttentionBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError


class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__(embed_dim, num_heads, dropout)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, E = x.size()
        k = self.k_proj(x)
        q = self.q_proj(x)
        v = self.v_proj(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(E)
        if mask is not None:
            scores = scores.masked_fill(mask[:, None, :], -1e9)
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        out = torch.matmul(weights, v)
        return self.out_proj(out)


class _HybridQuantumHead(nn.Module):
    def __init__(self, n_qubits: int, n_layers: int, device: Optional[object] = None):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.device = device or qml.device("default.qubit", wires=n_qubits)
        self.params = nn.Parameter(torch.randn(n_layers, n_qubits))
        def circuit(x, params):
            for l in range(n_layers):
                for q in range(n_qubits):
                    qml.RY(params[l, q], wires=q)
                for q in range(n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
                qml.CNOT(wires=[n_qubits - 1, 0])
            for q in range(n_qubits):
                qml.RY(x[q], wires=q)
            return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]
        self.qnode = qml.QNode(circuit, self.device, interface="torch")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H = x.shape
        out = torch.empty(B, T, self.n_qubits, device=x.device)
        for b in range(B):
            for t in range(T):
                inp = x[b, t, :self.n_qubits]
                out[b, t] = self.qnode(inp, self.params)
        if H > self.n_qubits:
            pad = torch.zeros(B, T, H - self.n_qubits, device=x.device)
            out = torch.cat([out, pad], dim=-1)
        return out


class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1,
                 n_qubits_per_head: int = 4, n_layers: int = 2, device: Optional[object] = None):
        super().__init__(embed_dim, num_heads, dropout)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.head_dim = embed_dim // num_heads
        self.quantum_heads = nn.ModuleList(
            [_HybridQuantumHead(n_qubits_per_head, n_layers, device) for _ in range(num_heads)]
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, E = x.size()
        k = self.k_proj(x)
        q = self.q_proj(x)
        v = self.v_proj(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(E)
        if mask is not None:
            scores = scores.masked_fill(mask[:, None, :], -1e9)
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        out = torch.matmul(weights, v)
        head_dim = E // self.num_heads
        out_heads = out.view(B, T, self.num_heads, head_dim)
        q_heads = []
        for i, head in enumerate(self.quantum_heads):
            q_heads.append(head(out_heads[:, :, i, :]))
        out = torch.stack(q_heads, dim=2).reshape(B, T, E)
        return self.out_proj(out)


class FeedForwardBase(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class FeedForwardClassical(FeedForwardBase):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.linear2 = nn.Linear(ffn_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class FeedForwardQuantum(FeedForwardBase):
    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int,
                 dropout: float = 0.1, device: Optional[object] = None):
        super().__init__(embed_dim, ffn_dim, dropout)
        self.n_qubits = n_qubits
        self.device = device or qml.device("default.qubit", wires=n_qubits)
        self.params = nn.Parameter(torch.randn(2, n_qubits))
        self.linear1 = nn.Linear(n_qubits, ffn_dim, bias=False)
        self.linear2 = nn.Linear(ffn_dim, embed_dim, bias=False)
        def circuit(x, params):
            for q in range(n_qubits):
                qml.RY(params[0, q], wires=q)
            for q in range(n_qubits - 1):
                qml.CNOT(wires=[q, q + 1])
            qml.CNOT(wires=[n_qubits - 1, 0])
            for q in range(n_qubits):
                qml.RY(x[q], wires=q)
            return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]
        self.qnode = qml.QNode(circuit, self.device, interface="torch")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.size()
        out = torch.empty(B, T, self.n_qubits, device=x.device)
        for b in range(B):
            for t in range(T):
                inp = x[b, t, :self.n_qubits]
                out[b, t] = self.qnode(inp, self.params)
        out = self.linear1(out)
        out = self.linear2(F.relu(self.dropout(out)))
        return out


class TransformerBlockBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TransformerBlockClassical(TransformerBlockBase):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class TransformerBlockQuantum(TransformerBlockBase):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 n_qubits_per_head: int, n_qubits_ffn: int,
                 dropout: float = 0.1, device: Optional[object] = None):
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout,
                                              n_qubits_per_head, device=device)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn,
                                      dropout, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TextClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int,
                 num_blocks: int, ffn_dim: int, num_classes: int,
                 dropout: float = 0.1, use_hybrid: bool = False,
                 n_qubits_per_head: int = 4, n_qubits_ffn: int = 4,
                 device: Optional[object] = None):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_enc = PositionalEncoder(embed_dim)
        if use_hybrid:
            blocks = [
                TransformerBlockQuantum(embed_dim, num_heads, ffn_dim,
                                       n_qubits_per_head, n_qubits_ffn,
                                       dropout, device=device)
                for _ in range(num_blocks)
            ]
        else:
            blocks = [
                TransformerBlockClassical(embed_dim, num_heads, ffn_dim,
                                          dropout)
                for _ in range(num_blocks)
            ]
        self.transformer = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_emb(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
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
