"""Quantum‑enhanced transformer implementation using TorchQuantum.

The module contains a hybrid transformer that can optionally use
quantum sub‑modules for attention and feed‑forward.  The quantum
implementations are based on TorchQuantum variational circuits
and are fully compatible with the classical API.  A lightweight
SelfAttentionQuantum helper is also provided, demonstrating how a
Qiskit‑style circuit can be wrapped to produce attention weights
in a classical workflow.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

# ---------------------------------------------------------------------------

class SelfAttentionQuantum:
    """Quantum self‑attention circuit using TorchQuantum.

    The circuit applies a parameterized RX rotation to each qubit
    followed by a chain of CNOTs that entangles the wires in a ring.
    The measurement in the Pauli‑Z basis yields expectation values
    that are interpreted as attention logits.
    """
    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.device = tq.QuantumDevice(n_wires=n_qubits)

        # Encoder: RX rotations parameterized by the input vector
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "rx", "wires": [i]}
                for i in range(n_qubits)
            ]
        )

        # Parameterized RX gates
        self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_qubits)])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def _build_circuit(self, qdev: tq.QuantumDevice, x: torch.Tensor) -> None:
        self.encoder(qdev, x)
        for wire, gate in enumerate(self.params):
            gate(qdev, wires=wire)
        for wire in range(self.n_qubits - 1):
            tqf.cnot(qdev, wires=[wire, wire + 1])
        tqf.cnot(qdev, wires=[self.n_qubits - 1, 0])

    def run(self, x: torch.Tensor) -> torch.Tensor:
        """Return a tensor of shape (batch, n_qubits) with expectation values."""
        qdev = self.device.copy(bsz=x.size(0), device=x.device)
        self._build_circuit(qdev, x)
        return self.measure(qdev)

# ---------------------------------------------------------------------------

class MultiHeadAttentionBase(nn.Module):
    """Shared interface for attention implementations."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout   = nn.Dropout(dropout)
        self.d_k = embed_dim // num_heads

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.size()
        return x.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)

    def attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                  mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        return torch.matmul(scores, value), scores

    def downstream(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                   mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = self.separate_heads(query)
        k = self.separate_heads(key)
        v = self.separate_heads(value)
        out, _ = self.attention(q, k, v, mask)
        return out.transpose(1, 2).contiguous().view(query.size(0), -1, self.embed_dim)

# ---------------------------------------------------------------------------

class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Standard multi‑head attention implemented with PyTorch."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.combine = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)
        return self.downstream(q, k, v, mask)

# ---------------------------------------------------------------------------

class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    """Quantum‑enhanced multi‑head attention.

    Each head projects the input into a quantum state using the
    `SelfAttentionQuantum` circuit and then performs a standard
    attention calculation on the resulting expectation values.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1,
                 n_qubits_per_head: int = 4) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.n_qubits = n_qubits_per_head
        self.head_circuits = nn.ModuleList([SelfAttentionQuantum(n_qubits_per_head) for _ in range(num_heads)])
        self.combine = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq, _ = x.size()
        # Quantum projection per head
        quantum_outputs = []
        for head_circ in self.head_circuits:
            # Project each token to a vector of length n_qubits
            proj = head_circ.run(x.view(-1, self.n_qubits))
            proj = proj.view(batch, seq, self.n_qubits)
            quantum_outputs.append(proj)
        quantum_tensor = torch.stack(quantum_outputs, dim=1)  # shape: (B, H, S, Nq)

        # Map quantum outputs back to embed_dim using a linear layer per head
        flat = quantum_tensor.view(batch, seq, -1)  # (B, S, H*Nq)
        projected = self.combine(flat)  # (B, S, embed_dim)

        return self.downstream(projected, projected, projected, mask)

# ---------------------------------------------------------------------------

class FeedForwardBase(nn.Module):
    """Base class for feed‑forward blocks."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim   = ffn_dim
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

# ---------------------------------------------------------------------------

class FeedForwardClassical(FeedForwardBase):
    """Standard two‑layer feed‑forward network."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

# ---------------------------------------------------------------------------

class FeedForwardQuantum(FeedForwardBase):
    """Variational feed‑forward network using a quantum circuit."""
    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int = 8, dropout: float = 0.1):
        super().__init__(embed_dim, ffn_dim, dropout)
        self.n_qubits = n_qubits
        self.qdev = tq.QuantumDevice(n_wires=n_qubits)
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
        )
        self.params = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)])
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def _run_qc(self, x: torch.Tensor) -> torch.Tensor:
        qdev = self.qdev.copy(bsz=x.size(0), device=x.device)
        self.encoder(qdev, x)
        for wire, gate in enumerate(self.params):
            gate(qdev, wires=wire)
        return self.measure(qdev)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, embed_dim)
        batch, seq, _ = x.size()
        proj = x.view(-1, self.n_qubits)
        q_out = self._run_qc(proj)
        q_out = q_out.view(batch, seq, self.n_qubits)
        out = self.linear1(self.dropout(q_out))
        return self.linear2(F.relu(out))

# ---------------------------------------------------------------------------

class TransformerBlockBase(nn.Module):
    """Base transformer block containing attention and feed‑forward."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1   = nn.LayerNorm(embed_dim)
        self.norm2   = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

# ---------------------------------------------------------------------------

class TransformerBlockClassical(TransformerBlockBase):
    """Pure‑classical transformer block."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

# ---------------------------------------------------------------------------

class TransformerBlockHybrid(TransformerBlockBase):
    """Hybrid transformer block that can use quantum sub‑modules."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 dropout: float = 0.1, use_quantum: bool = False,
                 n_qubits_per_head: int = 4, ffn_n_qubits: int = 8) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout,
                                              n_qubits_per_head) if use_quantum else MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, ffn_n_qubits, dropout) if use_quantum else FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------

class TextClassifier(nn.Module):
    """Quantum‑enhanced text classifier.

    The class accepts a ``use_quantum`` flag that, when set to True,
    replaces the classical attention and feed‑forward sub‑modules with
    their quantum counterparts.  The quantum implementations are
    built with TorchQuantum and expose the same API as the classical
    blocks, enabling a drop‑in replacement for hybrid experimentation.
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
        n_qubits_per_head: int = 4,
        ffn_n_qubits: int = 8,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding   = PositionalEncoder(embed_dim)
        block_cls = TransformerBlockHybrid if use_quantum else TransformerBlockClassical
        self.transformers = nn.Sequential(
            *[block_cls(embed_dim, num_heads, ffn_dim, dropout,
                       use_quantum, n_qubits_per_head, ffn_n_qubits)
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
    "SelfAttentionQuantum",
    "MultiHeadAttentionBase",
    "MultiHeadAttentionClassical",
    "MultiHeadAttentionQuantum",
    "FeedForwardBase",
    "FeedForwardClassical",
    "FeedForwardQuantum",
    "TransformerBlockBase",
    "TransformerBlockClassical",
    "TransformerBlockHybrid",
    "PositionalEncoder",
    "TextClassifier",
]
