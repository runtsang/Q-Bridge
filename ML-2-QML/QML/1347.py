"""Quantum‑enhanced transformer components implemented with Pennylane."""

from __future__ import annotations

import math
from typing import Optional, Iterable, Tuple, List

import pennylane as qml
import pennylane.numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Quantum primitives
# --------------------------------------------------------------------------- #

def _variational_circuit(params: np.ndarray, wires: List[int]) -> None:
    """Simple entangling circuit used for both attention and feed‑forward."""
    for i, w in enumerate(wires):
        qml.RX(params[0, i], wires=w)
    for i in range(len(wires) - 1):
        qml.CNOT(wires=[wires[i], wires[i + 1]])
    qml.CNOT(wires=[wires[-1], wires[0]])


# --------------------------------------------------------------------------- #
# Quantum attention module
# --------------------------------------------------------------------------- #

class QuantumAttention(nn.Module):
    """
    Quantum‑enhanced multi‑head attention.
    Each token is encoded into a quantum state, processed by a variational circuit,
    and the resulting expectation values are returned as new token embeddings.
    """

    def __init__(self, embed_dim: int, num_heads: int, n_qubits: int = 8, device: str = "default.qubit") -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.n_qubits = n_qubits
        self.head_dim = embed_dim // num_heads

        # Quantum device
        self.dev = qml.device(device, wires=n_qubits)

        # Trainable parameters for each head
        self.params = nn.Parameter(torch.randn(1, n_qubits))

        # Linear layers to map classical embeddings to quantum parameters
        self.encoder = nn.Linear(embed_dim, n_qubits, bias=False)

        # Linear layer to map quantum outputs back to embedding space
        self.decoder = nn.Linear(n_qubits, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq, _ = x.shape
        # Encode classical features into rotation angles
        angles = self.encoder(x)  # (B, S, n_qubits)
        out = torch.zeros_like(x)

        for b in range(batch):
            for s in range(seq):
                # Run the circuit for each token
                @qml.qnode(self.dev, interface="torch")
                def circuit(params):
                    _variational_circuit(params, range(self.n_qubits))
                    return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

                q_out = circuit(angles[b, s])
                out[b, s] = self.decoder(q_out)

        # Simple residual addition (classical skip connection)
        return out + x


# --------------------------------------------------------------------------- #
# Quantum feed‑forward module
# --------------------------------------------------------------------------- #

class QuantumFeedForward(nn.Module):
    """
    Quantum feed‑forward network that operates on each token independently.
    """

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int = 8, device: str = "default.qubit") -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.n_qubits = n_qubits

        # Quantum device
        self.dev = qml.device(device, wires=n_qubits)

        # Trainable parameters for the circuit
        self.params = nn.Parameter(torch.randn(1, n_qubits))

        # Linear layers to map between classical and quantum spaces
        self.encoder = nn.Linear(embed_dim, n_qubits, bias=False)
        self.decoder = nn.Linear(n_qubits, ffn_dim, bias=False)

        # Classical linear projection back to embedding dimension
        self.out_proj = nn.Linear(ffn_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        out = torch.zeros(batch, seq, self.ffn_dim)

        for b in range(batch):
            for s in range(seq):
                @qml.qnode(self.dev, interface="torch")
                def circuit(params):
                    _variational_circuit(params, range(self.n_qubits))
                    return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

                q_out = circuit(self.encoder(x[b, s]))
                out[b, s] = self.decoder(q_out)

        # Classical MLP on top of quantum output
        out = F.relu(out)
        out = self.out_proj(out)
        return out


# --------------------------------------------------------------------------- #
# Quantum transformer block
# --------------------------------------------------------------------------- #

class TransformerBlockQuantum(nn.Module):
    """
    Transformer block that uses quantum attention and feed‑forward sub‑modules.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        n_qubits_attn: int = 8,
        n_qubits_ffn: int = 8,
        dropout: float = 0.1,
        device: str = "default.qubit",
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.attn = QuantumAttention(embed_dim, num_heads, n_qubits_attn, device)
        self.ffn = QuantumFeedForward(embed_dim, ffn_dim, n_qubits_ffn, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


# --------------------------------------------------------------------------- #
# Positional encoding (identical to classical version)
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
# Quantum‑enhanced text classifier
# --------------------------------------------------------------------------- #

class TextClassifierQuantum(nn.Module):
    """
    Transformer‑based text classifier that uses fully quantum transformer blocks.
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
        n_qubits_attn: int = 8,
        n_qubits_ffn: int = 8,
        device: str = "default.qubit",
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        self.dropout = nn.Dropout(dropout)

        blocks = [
            TransformerBlockQuantum(
                embed_dim,
                num_heads,
                ffn_dim,
                n_qubits_attn,
                n_qubits_ffn,
                dropout,
                device,
            )
            for _ in range(num_blocks)
        ]
        self.transformers = nn.Sequential(*blocks)

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
    "QuantumAttention",
    "QuantumFeedForward",
    "TransformerBlockQuantum",
    "PositionalEncoder",
    "TextClassifierQuantum",
]
