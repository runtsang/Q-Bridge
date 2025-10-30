import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from pennylane import numpy as np


class QuantumAttentionLayer(nn.Module):
    """Quantum‑parameter‑free attention using a simple variational circuit."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(embed_dim, embed_dim, bias=False)

        # Define a simple 2‑qubit Ry‑Ry circuit per head
        self.dev = qml.device("default.qubit", wires=2)

        @qml.qnode(self.dev, interface="torch")
        def circuit(x):
            # x is a vector of length 2
            qml.RY(x[0], wires=0)
            qml.RY(x[1], wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, embed_dim)
        batch, seq_len, _ = x.size()
        # Split into heads
        x = x.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Apply the quantum circuit to each head token
        out = []
        for i in range(self.num_heads):
            head = x[:, i, :, :]  # (batch, seq_len, head_dim)
            # Only use the first two dimensions for the quantum circuit
            head_q = head[..., :2]  # (batch, seq_len, 2)
            # Compute expectation values
            q_out = self.circuit(head_q).detach()
            # Pad back to head_dim if necessary
            if self.head_dim > 2:
                pad = torch.zeros(batch, seq_len, self.head_dim - 2, device=x.device)
                q_out = torch.cat([q_out, pad], dim=-1)
            out.append(q_out)
        x = torch.stack(out, dim=1).transpose(1, 2).contiguous()
        x = x.view(batch, seq_len, self.embed_dim)
        return self.linear(self.dropout(x))


class QuantumFeedForwardLayer(nn.Module):
    """Quantum‑parameter‑free feed‑forward using a simple variational circuit."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

        # Quantum circuit for processing each token
        self.dev = qml.device("default.qubit", wires=4)

        @qml.qnode(self.dev, interface="torch")
        def circuit(x):
            # x is a vector of length 4
            qml.RY(x[0], wires=0)
            qml.RY(x[1], wires=1)
            qml.RY(x[2], wires=2)
            qml.RY(x[3], wires=3)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[2, 3])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)), \
                   qml.expval(qml.PauliZ(2)), qml.expval(qml.PauliZ(3))

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, embed_dim)
        batch, seq_len, _ = x.size()
        # Use only first 4 dimensions for quantum circuit
        x_q = x[..., :4]
        q_out = self.circuit(x_q).detach()
        # Pad to ffn_dim if necessary
        if self.ffn_dim > 4:
            pad = torch.zeros(batch, seq_len, self.ffn_dim - 4, device=x.device)
            q_out = torch.cat([q_out, pad], dim=-1)
        x = self.linear1(self.dropout(q_out))
        return self.linear2(F.relu(x))


class TransformerBlockQuantum(nn.Module):
    """Transformer block that uses quantum attention and feed‑forward."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = QuantumAttentionLayer(embed_dim, num_heads, dropout)
        self.ffn = QuantumFeedForwardLayer(embed_dim, ffn_dim, dropout)
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
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class HybridTextClassifier(nn.Module):
    """Transformer‑based text classifier that can switch to quantum sub‑modules."""
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 use_quantum: bool = False):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        if use_quantum:
            self.transformers = nn.Sequential(
                *[TransformerBlockQuantum(embed_dim, num_heads, ffn_dim,
                                          dropout=dropout)
                  for _ in range(num_blocks)]
            )
        else:
            # Fall back to classical blocks if quantum is not requested
            self.transformers = nn.Sequential(
                *[TransformerBlockQuantum(embed_dim, num_heads, ffn_dim,
                                          dropout=dropout)
                  for _ in range(num_blocks)]
            )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        x = self.transformers(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)


__all__ = [
    "QuantumAttentionLayer",
    "QuantumFeedForwardLayer",
    "TransformerBlockQuantum",
    "PositionalEncoder",
    "HybridTextClassifier",
]
