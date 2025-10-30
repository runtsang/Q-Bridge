"""Quantum‑enhanced transformer layers implemented with PennyLane."""

import math
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import pennylane.numpy as np


# --------------------------------------------------------------------------- #
# Positional encoder (identical to the classical version)
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Quantum layer that encodes a token into a parameterised circuit
# --------------------------------------------------------------------------- #
class QuantumLayer(nn.Module):
    """Simple quantum embedding that maps an embedding vector to a new one of the same size."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.q_device = qml.device("default.qubit", wires=embed_dim)
        # Trainable rotation angles
        self.weights = nn.Parameter(torch.randn(embed_dim))

    def quantum_forward(self, token: torch.Tensor) -> torch.Tensor:
        """Run a single token through the quantum circuit."""

        @qml.qnode(self.q_device, interface="torch")
        def circuit(wires, weights, token):
            # Encode input angles via RX
            for i in range(self.embed_dim):
                qml.RX(token[i], wires=wires[i])
            # Entangling layer
            for i in range(self.embed_dim - 1):
                qml.CNOT(wires[i], wires[i + 1])
            # Trainable rotations
            for i in range(self.embed_dim):
                qml.RZ(weights[i], wires=wires[i])
            # Return expectation values of Z
            return [qml.expval(qml.PauliZ(wires[i])) for i in range(self.embed_dim)]

        return circuit(self.embed_dim, self.weights, token)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the quantum circuit to every token."""
        B, T, E = x.shape
        out = torch.empty_like(x)
        for b in range(B):
            for t in range(T):
                out[b, t] = self.quantum_forward(x[b, t])
        return out


# --------------------------------------------------------------------------- #
# Quantum transformer block
# --------------------------------------------------------------------------- #
class QuantumTransformerBlock(nn.Module):
    """Transformer block whose core computation is performed on a quantum circuit."""

    def __init__(self, embed_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.quantum_layer = QuantumLayer(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_out = self.quantum_layer(x)
        x = self.norm1(x + self.dropout(q_out))
        x = self.norm2(x + self.dropout(q_out))
        return x


# --------------------------------------------------------------------------- #
# Quantum text classifier
# --------------------------------------------------------------------------- #
class QuantumTextClassifier(nn.Module):
    """Transformer‑based text classifier that relies entirely on quantum layers."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_blocks: int,
        num_classes: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        self.blocks = nn.ModuleList(
            [QuantumTransformerBlock(embed_dim, dropout) for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(
            embed_dim, num_classes if num_classes > 2 else 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)


__all__ = [
    "PositionalEncoder",
    "QuantumLayer",
    "QuantumTransformerBlock",
    "QuantumTextClassifier",
]
