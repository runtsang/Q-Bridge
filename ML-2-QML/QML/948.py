from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

# 4‑qubit device
dev = qml.device("default.qubit", wires=4)


@qml.qnode(dev, interface="torch")
def quantum_circuit(x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """Variational circuit that maps a 4‑dim vector to another 4‑dim vector."""
    # Encode input
    for i in range(4):
        qml.RZ(x[i], wires=i)
    # Variational ansatz
    for i in range(4):
        qml.RY(params[i], wires=i)
    # Entanglement
    for i in range(3):
        qml.CNOT(wires=[i, i + 1])
    qml.CNOT(wires=[3, 0])
    # Measurements
    return torch.stack([qml.expval(qml.PauliZ(i)) for i in range(4)])


class QuantumAttention(nn.Module):
    """Hybrid attention that uses a 4‑qubit variational circuit per head."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        if self.d_k!= 4:
            raise ValueError("QuantumAttention requires d_k=4 for the 4‑qubit circuit")
        self.dropout = nn.Dropout(dropout)
        # Linear projections
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.head_mix = nn.Linear(num_heads * self.d_k, embed_dim, bias=False)
        # Quantum parameters
        self.params = nn.Parameter(torch.randn(4))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq, _ = x.shape
        k = self.k_linear(x).view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        q = self.q_linear(x).view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(x).view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        # Apply quantum circuit to each head
        q_q = torch.stack([quantum_circuit(q[:, i, :], self.params) for i in range(self.num_heads)], dim=1)
        k_q = torch.stack([quantum_circuit(k[:, i, :], self.params) for i in range(self.num_heads)], dim=1)
        v_q = torch.stack([quantum_circuit(v[:, i, :], self.params) for i in range(self.num_heads)], dim=1)
        scores = torch.matmul(q_q, k_q.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v_q)
        out = out.transpose(1, 2).contiguous().view(batch, seq, self.embed_dim)
        return self.head_mix(out)


class QuantumFeedForward(nn.Module):
    """Feed‑forward network realised with a 4‑qubit variational circuit."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)
        # Map to quantum dimension
        self.linear1 = nn.Linear(embed_dim, 4, bias=False)
        self.linear2 = nn.Linear(4, ffn_dim, bias=False)
        self.out = nn.Linear(ffn_dim, embed_dim, bias=False)
        self.params = nn.Parameter(torch.randn(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        x_q = self.linear1(x).view(batch * seq, 4)
        q_out = torch.stack([quantum_circuit(x_q[i], self.params) for i in range(batch * seq)], dim=0)
        q_out = q_out.view(batch, seq, 4)
        x = self.linear2(q_out)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.out(x)
        return x


class TransformerBlockQuantum(nn.Module):
    """Quantum‑enhanced transformer block."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = QuantumAttention(embed_dim, num_heads, dropout)
        self.ffn = QuantumFeedForward(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class PositionalEncoder(nn.Module):
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
        return x + self.pe[:, : x.size(1)]


class QTransformerTorch__gen232(nn.Module):
    """Transformer‑based text classifier with quantum‑enhanced blocks."""
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        self.transformers = nn.Sequential(
            *[
                TransformerBlockQuantum(embed_dim, num_heads, ffn_dim, dropout)
                for _ in range(num_blocks)
            ]
        )
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
    "QuantumAttention",
    "QuantumFeedForward",
    "TransformerBlockQuantum",
    "PositionalEncoder",
    "QTransformerTorch__gen232",
]
