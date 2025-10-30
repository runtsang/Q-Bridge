"""Quantum‑centric transformer using Pennylane’s VQC to implement attention and feed‑forward."""

from __future__ import annotations

import math
from typing import Optional

import pennylane as qml
import pennylane.numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttentionBase(nn.Module):
    """Base class shared by all attention variants."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, use_bias: bool = False) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.attn_weights: Optional[torch.Tensor] = None

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        return x.view(batch, -1, self.num_heads, self.d_k).transpose(1, 2)

    def attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                  mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        probs = F.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        return torch.matmul(probs, value), probs

    def downstream(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                   batch_size: int, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = self.separate_heads(query)
        k = self.separate_heads(key)
        v = self.separate_heads(value)
        out, self.attn_weights = self.attention(q, k, v, mask)
        return out.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)


class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Standard multi‑head attention implemented with PyTorch."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, use_bias: bool = False) -> None:
        super().__init__(embed_dim, num_heads, dropout, use_bias)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.combine_heads = nn.Linear(embed_dim, embed_dim, bias=use_bias)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq, _ = x.shape
        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)
        out = self.downstream(q, k, v, batch, mask)
        return self.combine_heads(out)


class QAttentionLayer(nn.Module):
    """Quantum layer that transforms a vector of size d_k."""
    def __init__(self, n_qubits: int) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.qnode = qml.torch.QuantumNode(self.circuit, dev=self.dev, interface="torch")
        self.params = nn.Parameter(torch.randn(n_qubits))

    def circuit(self, inputs: torch.Tensor, params: torch.Tensor) -> list[torch.Tensor]:
        for i in range(self.n_qubits):
            qml.RY(inputs[i], wires=i)
        for i in range(self.n_qubits):
            qml.RY(params[i], wires=i)
        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        out = []
        for b in range(batch):
            batch_out = []
            for s in range(seq):
                out_vec = self.qnode(x[b, s], self.params)
                batch_out.append(out_vec)
            out.append(torch.stack(batch_out))
        return torch.stack(out)


class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    """Multi‑head attention where each head is a tiny quantum circuit."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1,
                 use_bias: bool = False) -> None:
        super().__init__(embed_dim, num_heads, dropout, use_bias)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.combine_heads = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.q_layers = nn.ModuleList([QAttentionLayer(self.d_k) for _ in range(num_heads)])

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq, _ = x.shape
        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)
        k = self.separate_heads(k)
        q = self.separate_heads(q)
        v = self.separate_heads(v)
        q_q = []
        k_q = []
        v_q = []
        for i in range(self.num_heads):
            q_head = q[:, :, i, :]
            k_head = k[:, :, i, :]
            v_head = v[:, :, i, :]
            q_head_q = self.q_layers[i](q_head)
            k_head_q = self.q_layers[i](k_head)
            v_head_q = self.q_layers[i](v_head)
            q_q.append(q_head_q)
            k_q.append(k_head_q)
            v_q.append(v_head_q)
        q = torch.stack(q_q, dim=2)
        k = torch.stack(k_q, dim=2)
        v = torch.stack(v_q, dim=2)
        out, _ = self.attention(q, k, v, mask)
        out = out.transpose(1, 2).contiguous().view(batch, seq, self.embed_dim)
        return self.combine_heads(out)


class FeedForwardBase(nn.Module):
    """Base class for feed‑forward layers."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)


class FeedForwardClassical(nn.Sequential):
    """Two‑layer MLP implemented with nn.Linear."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )


class QFFLayer(nn.Module):
    """Quantum layer that transforms a vector of size embed_dim."""
    def __init__(self, n_qubits: int) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.qnode = qml.torch.QuantumNode(self.circuit, dev=self.dev, interface="torch")
        self.params = nn.Parameter(torch.randn(n_qubits))

    def circuit(self, inputs: torch.Tensor, params: torch.Tensor) -> list[torch.Tensor]:
        for i in range(self.n_qubits):
            qml.RY(inputs[i], wires=i)
        for i in range(self.n_qubits):
            qml.RY(params[i], wires=i)
        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        out = []
        for b in range(batch):
            batch_out = []
            for s in range(seq):
                out_vec = self.qnode(x[b, s], self.params)
                batch_out.append(out_vec)
            out.append(torch.stack(batch_out))
        return torch.stack(out)


class FeedForwardQuantum(FeedForwardBase):
    """Feed‑forward that runs a quantum circuit per token."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.q_layer = QFFLayer(embed_dim)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.q_layer(x)
        out = self.linear1(self.dropout(out))
        out = F.relu(out)
        out = self.linear2(out)
        return out


class TransformerBlockBase(nn.Module):
    """Base transformer block containing attention and feed‑forward parts."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

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
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
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
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class QuantumPositionalEncoder(PositionalEncoder):
    """Quantum‑aware positional encoding using a simple VQC."""
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__(embed_dim, max_len)
        self.max_len = max_len
        self.n_qubits = embed_dim
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        self.qnode = qml.torch.QuantumNode(self.circuit, dev=self.dev, interface="torch")
        self.params = nn.Parameter(torch.randn(self.n_qubits))

    def circuit(self, pos: torch.Tensor, params: torch.Tensor) -> list[torch.Tensor]:
        angle = (pos / self.max_len) * math.pi
        for i in range(self.n_qubits):
            qml.RY(angle, wires=i)
        for i in range(self.n_qubits):
            qml.RY(params[i], wires=i)
        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        out = []
        for b in range(batch):
            batch_out = []
            for s in range(seq):
                pos = torch.tensor(s, dtype=torch.float32, device=x.device)
                out_vec = self.qnode(pos, self.params)
                batch_out.append(out_vec)
            out.append(torch.stack(batch_out))
        return torch.stack(out)


class TextClassifier(nn.Module):
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
        use_quantum_attention: bool = False,
        use_quantum_ffn: bool = False,
        use_quantum_positional_encoding: bool = False,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        if use_quantum_positional_encoding:
            self.pos_embedding = QuantumPositionalEncoder(embed_dim)
        else:
            self.pos_embedding = PositionalEncoder(embed_dim)
        block_cls = TransformerBlockQuantum if use_quantum_attention or use_quantum_ffn else TransformerBlockClassical
        self.transformers = nn.Sequential(
            *[block_cls(embed_dim, num_heads, ffn_dim, dropout) for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        if num_classes > 2:
            self.classifier = nn.Linear(embed_dim, num_classes)
        else:
            self.classifier = nn.Linear(embed_dim, 1)

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
    "QuantumPositionalEncoder",
    "TextClassifier",
]
