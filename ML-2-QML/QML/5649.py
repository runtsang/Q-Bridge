"""Quantum‑enhanced transformer using PennyLane.

This module provides the same public API as the classical version but
replaces the attention and feed‑forward sub‑modules with
parameterised quantum circuits.  The circuits are tiny variational
models that encode the input through RX rotations, apply a chain of
CNOTs and measure the Pauli‑Z expectation values.  The implementation
uses PennyLane’s Torch interface and is fully differentiable.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

# --------------------------------------------------------------------------- #
# Helper quantum layer
# --------------------------------------------------------------------------- #
class QuantumLayer(nn.Module):
    """Simple variational circuit that maps a vector of length ``n_qubits`` to
    another vector of the same length.  The circuit consists of an RX encoding
    of the input, a number of trainable RX layers, and a fixed CNOT entanglement
    pattern.  The output is the vector of Pauli‑Z expectation values.
    """

    def __init__(self, n_qubits: int, n_layers: int = 1):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)
        # trainable parameters for each layer
        self.params = nn.Parameter(torch.randn(n_layers, n_qubits))

    def _circuit(self, inputs, params):
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(inp, par):
            for i in range(self.n_qubits):
                qml.RX(inp[i], wires=i)
            for layer in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.RX(par[layer, i], wires=i)
                # simple entanglement pattern
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[self.n_qubits - 1, 0])
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        return circuit(inputs, params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, n_qubits)
        batch = x.shape[0]
        out = []
        for i in range(batch):
            out.append(self._circuit(x[i], self.params))
        return torch.stack(out)


# --------------------------------------------------------------------------- #
# Multi‑head attention
# --------------------------------------------------------------------------- #
class MultiHeadAttentionBase(nn.Module):
    """Base class for attention variants."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.head_dim = embed_dim // num_heads

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError


class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Standard multi‑head attention implemented entirely in PyTorch."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__(embed_dim, num_heads, dropout)
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq, _ = x.shape
        qkv = self.qkv(x).reshape(batch, seq, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        scores = torch.einsum("bshd,bshd->bhs", q, k) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum("bhs,bshd->bshd", attn, v)
        out = out.reshape(batch, seq, self.embed_dim)
        return self.out_proj(out)


class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    """Quantum‑augmented multi‑head attention."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, n_qubits: Optional[int] = None):
        super().__init__(embed_dim, num_heads, dropout)
        if n_qubits is None:
            n_qubits = self.head_dim
        self.n_qubits = n_qubits
        # classical linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        # quantum layers per head
        self.quantum_heads = nn.ModuleList([QuantumLayer(n_qubits) for _ in range(num_heads)])
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq, _ = x.shape
        q = self.q_proj(x).reshape(batch, seq, self.num_heads, self.n_qubits)
        k = self.k_proj(x).reshape(batch, seq, self.num_heads, self.n_qubits)
        v = self.v_proj(x).reshape(batch, seq, self.num_heads, self.n_qubits)
        # apply quantum layer per head
        q = torch.stack([self.quantum_heads[i](q[:, :, i, :]) for i in range(self.num_heads)], dim=2)
        k = torch.stack([self.quantum_heads[i](k[:, :, i, :]) for i in range(self.num_heads)], dim=2)
        v = torch.stack([self.quantum_heads[i](v[:, :, i, :]) for i in range(self.num_heads)], dim=2)
        # attention
        scores = torch.einsum("bshd,bshd->bhs", q, k) / math.sqrt(self.n_qubits)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum("bhs,bshd->bshd", attn, v)
        out = out.reshape(batch, seq, self.embed_dim)
        return self.out_proj(out)


# --------------------------------------------------------------------------- #
# Feed‑forward network
# --------------------------------------------------------------------------- #
class FeedForwardBase(nn.Module):
    """Base class for feed‑forward sub‑modules."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class FeedForwardClassical(FeedForwardBase):
    """Two‑layer perceptron feed‑forward network."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class FeedForwardQuantum(FeedForwardBase):
    """Feed‑forward network realised by a quantum module."""

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: Optional[int] = None, dropout: float = 0.1):
        super().__init__(embed_dim, ffn_dim, dropout)
        if n_qubits is None:
            n_qubits = ffn_dim
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.quantum_layer = QuantumLayer(n_qubits)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.quantum_layer(x)
        return self.linear2(F.relu(x))


# --------------------------------------------------------------------------- #
# Transformer block
# --------------------------------------------------------------------------- #
class TransformerBlockBase(nn.Module):
    """Base transformer block containing attention and feed‑forward parts."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TransformerBlockQuantum(TransformerBlockBase):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        n_qubits_attn: Optional[int] = None,
        n_qubits_ffn: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionQuantum(
            embed_dim, num_heads, dropout, n_qubits_attn
        )
        self.ffn = FeedForwardQuantum(
            embed_dim, ffn_dim, n_qubits_ffn, dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class TransformerBlockClassical(TransformerBlockBase):
    """Fallback classical block used when quantum flag is False."""

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


# --------------------------------------------------------------------------- #
# Positional encoding
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Main classifier
# --------------------------------------------------------------------------- #
class QTransformerTorch__gen021(nn.Module):
    """Hybrid transformer classifier with optional quantum blocks.

    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary.
    embed_dim : int
        Dimensionality of token embeddings.
    num_heads : int
        Number of attention heads.
    num_blocks : int
        Number of transformer blocks.
    ffn_dim : int
        Feed‑forward hidden dimension.
    num_classes : int
        Number of output classes.
    dropout : float, optional
        Drop‑out probability.
    quantum : bool, optional
        If True, each block uses the quantum implementation.  When
        ``quantum=False`` the classical block is used.
    n_qubits_attn : int, optional
        Number of qubits per head in the quantum attention module.
    n_qubits_ffn : int, optional
        Number of qubits in the quantum feed‑forward module.
    """

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
        n_qubits_attn: Optional[int] = None,
        n_qubits_ffn: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        block_cls = TransformerBlockQuantum if quantum else TransformerBlockClassical
        self.transformers = nn.Sequential(
            *[
                block_cls(
                    embed_dim,
                    num_heads,
                    ffn_dim,
                    n_qubits_attn,
                    n_qubits_ffn,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(
            embed_dim, num_classes if num_classes > 2 else 1
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
    "MultiHeadAttentionQuantum",
    "MultiHeadAttentionClassical",
    "FeedForwardBase",
    "FeedForwardQuantum",
    "FeedForwardClassical",
    "TransformerBlockBase",
    "TransformerBlockQuantum",
    "TransformerBlockClassical",
    "PositionalEncoder",
    "QTransformerTorch__gen021",
]
