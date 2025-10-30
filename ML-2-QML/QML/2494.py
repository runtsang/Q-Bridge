"""Quantum‑enhanced transformer for use with TorchQuantum.

This module exposes the same public API as the classical
`UnifiedSelfAttentionTransformer` but replaces the attention and
feed‑forward sub‑modules with variational quantum circuits.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

# Quantum attention
class MultiHeadAttentionQuantum(nn.Module):
    class _QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.parameters = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for wire, gate in enumerate(self.parameters):
                gate(q_device, wires=wire)
            for wire in range(self.n_wires - 1):
                tqf.cnot(q_device, wires=[wire, wire + 1])
            tqf.cnot(q_device, wires=[self.n_wires - 1, 0])
            return self.measure(q_device)

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.1,
                 q_device: Optional[tq.QuantumDevice] = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.q_layers = nn.ModuleList(
            [self._QLayer(self.d_k) for _ in range(num_heads)]
        )
        self.q_device = q_device or tq.QuantumDevice(n_wires=self.d_k)
        self.combine = nn.Linear(embed_dim, embed_dim)

    def _projection(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        proj = torch.zeros(batch, seq, self.num_heads, self.d_k,
                           device=x.device, dtype=x.dtype)
        for b in range(batch):
            for s in range(seq):
                vec = x[b, s].view(1, -1)
                head_vecs = vec.reshape(1, self.num_heads, self.d_k)
                for h in range(self.num_heads):
                    qdev = self.q_device.copy(bsz=vec.size(0), device=vec.device)
                    proj[b, s, h] = self.q_layers[h](head_vecs[:, h, :], qdev)
        return proj.view(batch, seq, self.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self._projection(x)
        k = self._projection(x)
        v = self._projection(x)
        scores = F.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.d_k), dim=-1)
        return self.combine(self.dropout(scores @ v))

# Quantum feed‑forward
class FeedForwardQuantum(nn.Module):
    class _QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int):
            super().__init__()
            self.n_qubits = n_qubits
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
            )
            self.parameters = nn.ModuleList(
                [tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for wire, gate in enumerate(self.parameters):
                gate(q_device, wires=wire)
            return self.measure(q_device)

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1):
        super().__init__()
        self.q_layer = self._QLayer(n_qubits)
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = []
        for token in x.unbind(dim=1):
            qdev = self.q_device.copy(bsz=token.size(0), device=token.device)
            out.append(self.q_layer(token, qdev))
        out = torch.stack(out, dim=1)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))

# Positional encoding (identical to classical)
class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) *
                             (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]

# Unified quantum transformer
class UnifiedSelfAttentionTransformer(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 n_qubits_transformer: int = 0,
                 n_qubits_ffn: int = 0):
        super().__init__()
        if n_qubits_transformer <= 0:
            raise ValueError("Quantum transformer requires at least one qubit.")
        self.transformer = nn.Sequential(
            *[self._make_block(embed_dim, num_heads, ffn_dim,
                               dropout, n_qubits_transformer, n_qubits_ffn)
              for _ in range(num_blocks)]
        )
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim,
                                    num_classes if num_classes > 2 else 1)

    def _make_block(self, embed_dim, num_heads, ffn_dim, dropout,
                    n_qubits_transformer, n_qubits_ffn):
        block = nn.ModuleDict()
        block["attn"] = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout)
        block["ffn"] = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, dropout)
        block["norm1"] = nn.LayerNorm(embed_dim)
        block["norm2"] = nn.LayerNorm(embed_dim)
        block["dropout"] = nn.Dropout(dropout)
        return block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        x = self.pos_encoder(x)
        for block in self.transformer:
            attn_out = block["attn"](x)
            x = block["norm1"](x + block["dropout"](attn_out))
            ffn_out = block["ffn"](x)
            x = block["norm2"](x + block["dropout"](ffn_out))
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)

__all__ = [
    "MultiHeadAttentionQuantum",
    "FeedForwardQuantum",
    "PositionalEncoder",
    "UnifiedSelfAttentionTransformer",
]
