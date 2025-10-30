"""
Quantum‑Enhanced Transformer – quantum implementation using Pennylane.

This module mirrors the classical API but replaces the attention and feed‑forward
blocks with variational circuits.  The circuits are defined with Pennylane’s
`qnode` interface, enabling end‑to‑end autograd during training.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml


@dataclass
class QuantumConfig:
    """Configuration for quantum circuits."""
    n_qubits: int = 4
    layers: int = 1
    backend: str = "default.qubit"


class QAttentionHead(nn.Module):
    """Single attention head implemented as a variational circuit."""
    def __init__(self, n_qubits: int, layers: int, backend: str) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.layers = layers
        self.dev = qml.device(backend, wires=n_qubits)
        self.params = nn.Parameter(torch.randn(layers, n_qubits, 3))

        @qml.qnode(self.dev, interface="torch")
        def circuit(x, params):
            # Encode input vector into rotations
            for i in range(n_qubits):
                qml.RY(x[i], wires=i)
            # Variational layers
            for l in range(layers):
                for i in range(n_qubits):
                    qml.RX(params[l, i, 0], wires=i)
                    qml.RY(params[l, i, 1], wires=i)
                    qml.RZ(params[l, i, 2], wires=i)
                # Entangle neighbours
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[n_qubits - 1, 0])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, n_qubits)
        return self.circuit(x, self.params)


class QFeedForward(nn.Module):
    """Feed‑forward module realised by a variational circuit."""
    def __init__(self, n_qubits: int, layers: int, backend: str) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.layers = layers
        self.dev = qml.device(backend, wires=n_qubits)
        self.params = nn.Parameter(torch.randn(layers, n_qubits, 3))

        @qml.qnode(self.dev, interface="torch")
        def circuit(x, params):
            for i in range(n_qubits):
                qml.RY(x[i], wires=i)
            for l in range(layers):
                for i in range(n_qubits):
                    qml.RX(params[l, i, 0], wires=i)
                    qml.RY(params[l, i, 1], wires=i)
                    qml.RZ(params[l, i, 2], wires=i)
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[n_qubits - 1, 0])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.circuit(x, self.params)


class MultiHeadAttentionQuantum(nn.Module):
    """Multi‑head attention where each head is a quantum circuit."""
    def __init__(self, embed_dim: int, num_heads: int, qconfig: QuantumConfig) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.heads = nn.ModuleList(
            [QAttentionHead(qconfig.n_qubits, qconfig.layers, qconfig.backend) for _ in range(num_heads)]
        )
        self.out_proj = nn.Linear(num_heads * qconfig.n_qubits, embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq, _ = x.shape
        heads_input = x.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        out = []
        for i, head in enumerate(self.heads):
            head_in = heads_input[:, i, :, :].reshape(batch * seq, self.d_k)
            head_out = head(head_in)
            head_out = head_out.reshape(batch, seq, -1)
            out.append(head_out)
        concat = torch.cat(out, dim=-1)
        return self.out_proj(concat)


class FeedForwardQuantum(nn.Module):
    """Feed‑forward implemented with a quantum circuit."""
    def __init__(self, embed_dim: int, ffn_dim: int, qconfig: QuantumConfig) -> None:
        super().__init__()
        self.quantum = QFeedForward(qconfig.n_qubits, qconfig.layers, qconfig.backend)
        self.fc = nn.Linear(qconfig.n_qubits, ffn_dim)
        self.out = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        # For simplicity, use the mean over the sequence for the quantum input
        x_mean = x.mean(dim=1)
        x_reduced = x_mean[:, :self.quantum.n_qubits]
        q_out = self.quantum(x_reduced)
        ffn_out = self.fc(q_out)
        return self.out(ffn_out)


class TransformerBlockQuantum(nn.Module):
    """Transformer block built from quantum attention and feed‑forward."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        qconfig: QuantumConfig,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, qconfig)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, qconfig)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoder."""
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


class TextClassifier(nn.Module):
    """Quantum transformer‑based classifier."""
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        qconfig: QuantumConfig = QuantumConfig(),
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.blocks = nn.Sequential(
            *[TransformerBlockQuantum(embed_dim, num_heads, ffn_dim, qconfig, dropout)
              for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 1 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        x = self.pos_encoder(x)
        x = self.blocks(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)


__all__ = [
    "QuantumConfig",
    "QAttentionHead",
    "QFeedForward",
    "MultiHeadAttentionQuantum",
    "FeedForwardQuantum",
    "TransformerBlockQuantum",
    "PositionalEncoder",
    "TextClassifier",
]
