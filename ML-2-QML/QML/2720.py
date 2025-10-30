"""Quantum‑enhanced transformer with Qiskit circuits.

The implementation mirrors the classical version but replaces
self‑attention and feed‑forward sub‑modules with simple Qiskit
variational circuits.  The API remains identical so that the same
training loop can be used with either classical or quantum back‑ends.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import QuantumCircuit, execute, Aer

# Positional encoding (identical to the classical version)
class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-np.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

# Quantum self‑attention
class QuantumSelfAttention(nn.Module):
    """Self‑attention block realised with a Qiskit circuit."""
    def __init__(self, embed_dim: int, backend=None, shots: int = 1024):
        super().__init__()
        self.embed_dim = embed_dim
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        out = torch.zeros(batch, seq, self.embed_dim, device=x.device, dtype=x.dtype)
        for b in range(batch):
            for s in range(seq):
                qc = QuantumCircuit(self.embed_dim)
                # Random rotations for demonstration; in practice use learnable params
                for i in range(self.embed_dim):
                    qc.rx(np.random.rand() * 2 * np.pi, i)
                for i in range(self.embed_dim - 1):
                    qc.cx(i, i + 1)
                qc.measure_all()
                job = execute(qc, self.backend, shots=self.shots)
                counts = job.result().get_counts()
                probs = np.zeros(self.embed_dim)
                for bitstring, cnt in counts.items():
                    for i, bit in enumerate(bitstring[::-1]):  # reverse to match qubit order
                        probs[i] += cnt * int(bit)
                probs = probs / self.shots
                out[b, s] = torch.tensor(probs, device=x.device, dtype=x.dtype)
        return out

# Quantum feed‑forward
class FeedForwardQuantum(nn.Module):
    """Feed‑forward network realised by a Qiskit circuit."""
    def __init__(self, embed_dim: int, ffn_dim: int, backend=None, shots: int = 1024):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.linear = nn.Linear(embed_dim, ffn_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        out = torch.zeros(batch, seq, self.ffn_dim, device=x.device, dtype=x.dtype)
        for b in range(batch):
            for s in range(seq):
                qc = QuantumCircuit(self.embed_dim)
                for i in range(self.embed_dim):
                    qc.rx(np.random.rand() * 2 * np.pi, i)
                for i in range(self.embed_dim - 1):
                    qc.cx(i, i + 1)
                qc.measure_all()
                job = execute(qc, self.backend, shots=self.shots)
                counts = job.result().get_counts()
                probs = np.zeros(self.embed_dim)
                for bitstring, cnt in counts.items():
                    for i, bit in enumerate(bitstring[::-1]):
                        probs[i] += cnt * int(bit)
                probs = probs / self.shots
                probs_tensor = torch.tensor(probs, device=x.device, dtype=x.dtype)
                out[b, s] = self.linear(probs_tensor)
        return out

# Quantum transformer block
class TransformerBlockQuantum(nn.Module):
    """Transformer block with quantum self‑attention and feed‑forward."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.attn = QuantumSelfAttention(embed_dim)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

# Hybrid transformer model (quantum‑enabled)
class HybridSelfAttentionTransformer(nn.Module):
    """Unified transformer that can run in classical or quantum mode."""
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        use_quantum: bool = True,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.use_quantum = use_quantum

        if use_quantum:
            blocks = [
                TransformerBlockQuantum(embed_dim, num_heads, ffn_dim, dropout)
                for _ in range(num_blocks)
            ]
        else:
            blocks = [
                TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
                for _ in range(num_blocks)
            ]
        self.transformer = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_encoder(tokens)
        x = self.transformer(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)

__all__ = ["HybridSelfAttentionTransformer"]
