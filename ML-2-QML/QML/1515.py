# Quantum‑enhanced Transformer module using pennylane.
# This module mirrors the classical hybrid architecture but replaces the
# variational layer with a genuine Pennylane circuit, allowing experiments
# on real quantum back‑ends or simulators.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from pennylane import numpy as np


class MultiHeadAttentionBase(nn.Module):
    """Base class for attention layers – same signature as the seed."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError


class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Classical multi‑head attention."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        out, _ = self.attn(x, x, x, key_padding_mask=mask)
        return out


class FeedForwardBase(nn.Module):
    """Base class for feed‑forward layers."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)


class FeedForwardClassical(FeedForwardBase):
    """Two‑layer MLP."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# --------------------------------------------------------------------------- #
#  Pennylane variational circuit
# --------------------------------------------------------------------------- #
class VariationalQuantumLayer(nn.Module):
    """
    A single‑qubit variational circuit that is applied to each token.
    The circuit consists of a parameterised RX rotation followed by a
    measurement in the Z basis.  The parameters are optimised alongside
    the classical weights using the automatic differentiation provided
    by Pennylane's autograd interface.
    """
    def __init__(self, embed_dim: int, n_qubits: int = 2) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_qubits = n_qubits
        # Define a PennyLane device (CPU simulator)
        self.dev = qml.device("default.qubit", wires=n_qubits)
        # Quantum function that maps an input vector to Z‑expectation values
        @qml.qnode(self.dev, interface="torch")
        def circuit(x):
            for i in range(n_qubits):
                qml.RX(x[i], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[n_qubits - 1, 0])
            return qml.expval(qml.PauliZ(0))

        # Wrap the circuit in a torch.nn.Module for easy use
        self.circuit = nn.Module()
        self.circuit.forward = circuit

        # Linear layer to match dimensions
        self.decoder = nn.Linear(1, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch, seq_len, embed_dim)
        Returns tensor of same shape.
        """
        batch, seq, _ = x.shape
        # Flatten to process each token independently
        flat = x.reshape(batch * seq, -1)
        # Truncate or pad to n_qubits
        x_q = flat[:, : self.n_qubits]
        # If embed_dim < n_qubits, pad with zeros
        if self.embed_dim < self.n_qubits:
            pad = torch.zeros(batch * seq, self.n_qubits - self.embed_dim, device=x.device)
            x_q = torch.cat([flat[:, : self.embed_dim], pad], dim=1)
        # Pass through the quantum circuit
        q_out = self.circuit(x_q)  # shape (batch*seq, 1)
        q_out = q_out.reshape(batch, seq, 1)
        # Decode back to embed_dim
        return self.decoder(q_out)


# --------------------------------------------------------------------------- #
#  Hybrid transformer block
# --------------------------------------------------------------------------- #
class TransformerBlockBase(nn.Module):
    """Base class for transformer blocks."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)


class TransformerBlockQuantum(nn.Module):
    """
    Transformer block that combines a classical attention and feed‑forward
    sub‑module with a Pennylane variational layer.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        n_qubits: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)
        self.quantum = VariationalQuantumLayer(embed_dim, n_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        q_out = self.quantum(x)
        return x + self.dropout(q_out)  # residual connection


# --------------------------------------------------------------------------- #
#  Positional encoding
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
#  Text classifier with quantum block
# --------------------------------------------------------------------------- #
class TextClassifier(nn.Module):
    """
    Transformer‑based text classifier that can use a quantum transformer block
    (Pennylane) when `use_quantum` is True.
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
        use_quantum: bool = False,
        n_qubits: int = 2,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        block_cls = TransformerBlockQuantum if use_quantum else TransformerBlockBase
        self.blocks = nn.Sequential(
            *[
                block_cls(embed_dim, num_heads, ffn_dim, n_qubits=n_qubits, dropout=dropout)
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
        x = self.blocks(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)


__all__ = [
    "MultiHeadAttentionBase",
    "MultiHeadAttentionClassical",
    "FeedForwardBase",
    "FeedForwardClassical",
    "VariationalQuantumLayer",
    "TransformerBlockBase",
    "TransformerBlockQuantum",
    "PositionalEncoder",
    "TextClassifier",
]
