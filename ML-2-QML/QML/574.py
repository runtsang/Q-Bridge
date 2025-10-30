"""Hybrid transformer classifier with quantum sub‑modules.

This module implements the same public API as the classical version but
replaces the attention and feed‑forward layers with lightweight quantum
variational circuits.  The depth of the quantum circuits is controlled
by the ``quantum_depth`` argument, allowing a user to trade off
expressivity for simulation cost.

Key quantum features:
* Multi‑head attention is realised by a tiny parameter‑efficient
  variational circuit per head.
* Feed‑forward network is a hybrid quantum‑classical block that
  encodes the token vector into qubits, runs a depth‑controlled
  variational circuit, and projects back to a classical vector.
* The quantum depth can be set to zero to recover the fully classical
  behaviour, making the class a drop‑in replacement for the seed
  implementation.
"""

from __future__ import annotations

import math
from typing import Optional, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class MultiHeadAttentionBase(nn.Module):
    """Base class for attention layers, providing utilities."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.size()
        return x.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)

    def attention(self,
                  query: torch.Tensor,
                  key: torch.Tensor,
                  value: torch.Tensor,
                  mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        return self.dropout(attn), self.dropout(scores)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    """Quantum‑enhanced multi‑head attention."""

    class QLayer(tq.QuantumModule):
        def __init__(self, d_k: int, depth: int):
            super().__init__()
            self.n_wires = d_k
            self.depth = depth
            self.params_rx = nn.ParameterList(
                [nn.Parameter(torch.randn(d_k)) for _ in range(depth)]
            )
            self.params_rz = nn.ParameterList(
                [nn.Parameter(torch.randn(d_k)) for _ in range(depth)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            for wire in range(self.n_wires):
                tqf.rx(q_device, wires=[wire], params=x[:, wire])
            for l in range(self.depth):
                for wire in range(self.n_wires):
                    tqf.rx(q_device, wires=[wire], params=self.params_rx[l][wire])
                    tqf.rz(q_device, wires=[wire], params=self.params_rz[l][wire])
                for wire in range(self.n_wires - 1):
                    tqf.cnot(q_device, wires=[wire, wire + 1])
                tqf.cnot(q_device, wires=[self.n_wires - 1, 0])
            return self.measure(q_device)

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.0,
                 quantum_depth: int = 1,
                 q_device: Optional[tq.QuantumDevice] = None):
        super().__init__(embed_dim, num_heads, dropout)
        self.quantum_depth = quantum_depth
        self.q_layer = self.QLayer(self.d_k, quantum_depth)
        self.q_device = q_device
        self.combine_heads = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq, _ = x.size()
        x_heads = self.separate_heads(x).contiguous()
        out_heads = torch.zeros(batch, self.num_heads, seq, self.d_k,
                                device=x.device, dtype=x.dtype)
        for b in range(batch):
            for h in range(self.num_heads):
                head_vec = x_heads[b, h]
                token_out = []
                for t in range(seq):
                    token = head_vec[t].unsqueeze(0)
                    qdev = self.q_device or tq.QuantumDevice(
                        n_wires=self.n_wires, bsz=1, device=token.device
                    )
                    out = self.q_layer(token, qdev)
                    token_out.append(out.squeeze(0))
                out_heads[b, h] = torch.stack(token_out, dim=0)
        out = out_heads.transpose(1, 2).contiguous().view(batch, seq, self.embed_dim)
        return self.combine_heads(out)


class FeedForwardBase(nn.Module):
    """Base class for feed‑forward layers."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class FeedForwardQuantum(FeedForwardBase):
    """Hybrid quantum‑classical feed‑forward block."""

    class QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int, depth: int):
            super().__init__()
            self.n_wires = n_qubits
            self.depth = depth
            self.params_rx = nn.ParameterList(
                [nn.Parameter(torch.randn(n_qubits)) for _ in range(depth)]
            )
            self.params_rz = nn.ParameterList(
                [nn.Parameter(torch.randn(n_qubits)) for _ in range(depth)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            for wire in range(self.n_wires):
                tqf.rx(q_device, wires=[wire], params=x[:, wire])
            for l in range(self.depth):
                for wire in range(self.n_wires):
                    tqf.rx(q_device, wires=[wire], params=self.params_rx[l][wire])
                    tqf.rz(q_device, wires=[wire], params=self.params_rz[l][wire])
                for wire in range(self.n_wires - 1):
                    tqf.cnot(q_device, wires=[wire, wire + 1])
                tqf.cnot(q_device, wires=[self.n_wires - 1, 0])
            return self.measure(q_device)

    def __init__(self,
                 embed_dim: int,
                 ffn_dim: int,
                 quantum_depth: int = 1,
                 n_qubits: int = 8,
                 dropout: float = 0.0):
        super().__init__(embed_dim, ffn_dim, dropout)
        self.quantum_depth = quantum_depth
        self.n_qubits = min(embed_dim, n_qubits)
        self.q_layer = self.QLayer(self.n_qubits, quantum_depth)
        self.q_device = tq.QuantumDevice(n_wires=self.n_qubits)
        self.linear1 = nn.Linear(self.n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.size()
        out = torch.zeros(batch, seq, self.n_qubits,
                          device=x.device, dtype=x.dtype)
        for b in range(batch):
            for t in range(seq):
                token = x[b, t].unsqueeze(0)
                q_input = token[:, :self.n_qubits]
                qdev = self.q_device.copy(bsz=1, device=token.device)
                q_out = self.q_layer(q_input, qdev)
                out[b, t] = q_out.squeeze(0)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))


class TransformerBlockBase(nn.Module):
    """Base transformer block."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TransformerBlockClassical(TransformerBlockBase):
    """Classic transformer block used when quantum_depth is zero."""

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 dropout: float = 0.0,
                 block_idx: int = 0,
                 num_blocks: int = 1):
        super().__init__(embed_dim, num_heads, dropout)
        self.dropout = nn.Dropout(dropout * (1 + block_idx / num_blocks))
        self.attn = MultiHeadAttentionClassical(embed_dim,
                                                num_heads,
                                                dropout=self.dropout.p)
        self.ffn = FeedForwardClassical(embed_dim,
                                        ffn_dim,
                                        dropout=self.dropout.p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class TransformerBlockQuantum(TransformerBlockBase):
    """Quantum‑enhanced transformer block."""

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 quantum_depth: int,
                 n_qubits: int,
                 dropout: float = 0.0,
                 block_idx: int = 0,
                 num_blocks: int = 1):
        super().__init__(embed_dim, num_heads, dropout)
        self.dropout = nn.Dropout(dropout * (1 + block_idx / num_blocks))
        self.attn = MultiHeadAttentionQuantum(embed_dim,
                                              num_heads,
                                              dropout=self.dropout.p,
                                              quantum_depth=quantum_depth)
        self.ffn = FeedForwardQuantum(embed_dim,
                                      ffn_dim,
                                      quantum_depth=quantum_depth,
                                      n_qubits=n_qubits,
                                      dropout=self.dropout.p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class QuantumTransformerAdapter(nn.Module):
    """
    Hybrid transformer classifier that can run entirely classically or
    activate quantum sub‑modules via the ``quantum_depth`` parameter.
    The API matches that of the original TextClassifier.
    """

    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 quantum_depth: int = 0,
                 n_qubits: int = 8,
                 *_, **__):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        blocks = nn.ModuleList()
        for i in range(num_blocks):
            if quantum_depth > 0:
                block = TransformerBlockQuantum(embed_dim,
                                                num_heads,
                                                ffn_dim,
                                                quantum_depth=quantum_depth,
                                                n_qubits=n_qubits,
                                                dropout=dropout,
                                                block_idx=i,
                                                num_blocks=num_blocks)
            else:
                block = TransformerBlockClassical(embed_dim,
                                                  num_heads,
                                                  ffn_dim,
                                                  dropout=dropout,
                                                  block_idx=i,
                                                  num_blocks=num_blocks)
            blocks.append(block)
        self.blocks = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim,
                                    num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        x = self.blocks(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)


__all__ = [
    "MultiHeadAttentionBase",
    "MultiHeadAttentionQuantum",
    "FeedForwardBase",
    "FeedForwardQuantum",
    "TransformerBlockBase",
    "TransformerBlockQuantum",
    "PositionalEncoder",
    "QuantumTransformerAdapter",
]
