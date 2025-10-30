"""
Quantum‑enhanced transformer that replaces the feed‑forward sub‑module
with a variational quantum circuit.  The attention remains classical
to keep the training stable, while the quantum feed‑forward network
provides a small quantum sub‑module that can be swapped in or out.
"""

from __future__ import annotations

import math
from typing import Optional

import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000):
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


class QuantumFeedForward(nn.Module):
    """
    Variational quantum feed‑forward network.
    Uses a small PennyLane circuit per token.
    """
    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        qubits: int = 2,
        n_layers: int = 1,
    ):
        super().__init__()
        if ffn_dim % qubits!= 0:
            raise ValueError("ffn_dim must be divisible by qubits")
        self.qubits = qubits
        self.n_layers = n_layers
        self.readout_wires = list(range(qubits))
        self.params = nn.Parameter(
            torch.randn(n_layers, qubits, 3)
        )  # 3 rotation angles per qubit per layer
        self.backend = "default.qubit"
        self.device = qml.device(self.backend, wires=self.readout_wires)
        self.linear1 = nn.Linear(qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def _qnode(self, inputs: torch.Tensor) -> torch.Tensor:
        @qml.qnode(self.device, interface="torch")
        def circuit(x):
            # encode input as rotations on each qubit
            for w in range(self.qubits):
                qml.RX(x[w], wires=w)
            for layer in range(self.n_layers):
                qml.StronglyEntanglingLayers(self.params[layer], wires=self.readout_wires)
            return [qml.expval(qml.PauliZ(w)) for w in self.readout_wires]

        return circuit(inputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, embed_dim)
        batch, seq, _ = x.size()
        out = []
        for i in range(batch):
            batch_out = []
            for j in range(seq):
                token = x[i, j]
                # take first qubits dimensions as input to the circuit
                inp = token[: self.qubits]
                qout = self._qnode(inp)
                batch_out.append(qout)
            out.append(torch.stack(batch_out, dim=0))
        out = torch.stack(out, dim=0)  # (batch, seq, qubits)
        out = self.linear1(out)
        out = F.relu(out)
        out = self.linear2(out)
        return out


class TransformerBlockQuantum(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        qubits: int = 2,
        n_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = QuantumFeedForward(embed_dim, ffn_dim, qubits, n_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        return self.norm2(x + ffn_out)


class UnifiedQTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        qubits: int = 2,
        n_layers: int = 1,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlockQuantum(
                    embed_dim, num_heads, ffn_dim,
                    qubits=qubits, n_layers=n_layers, dropout=dropout,
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
        x = self.pos_encoder(tokens)
        for block in self.transformer_blocks:
            x = block(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)


__all__ = [
    "PositionalEncoder",
    "QuantumFeedForward",
    "TransformerBlockQuantum",
    "UnifiedQTransformer",
]
