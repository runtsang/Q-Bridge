"""UnifiedSelfAttentionTransformer – classical core with optional quantum sub‑modules."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

# --------------------------------------------------------------------------- #
# Base attention logic
# --------------------------------------------------------------------------- #
class _BaseAttention(nn.Module):
    """Shared attention machinery for classical and quantum variants."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        return x.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)

    def attention(self, q, k, v, mask: Optional[torch.Tensor] = None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = self.dropout(F.softmax(scores, dim=-1))
        return torch.matmul(scores, v), scores

    def downstream(self, q, k, v, mask: Optional[torch.Tensor] = None):
        qh = self.separate_heads(q)
        kh = self.separate_heads(k)
        vh = self.separate_heads(v)
        out, attn = self.attention(qh, kh, vh, mask)
        return out.transpose(1, 2).contiguous().view(*q.shape), attn


# --------------------------------------------------------------------------- #
# Classical attention
# --------------------------------------------------------------------------- #
class ClassicalAttention(_BaseAttention):
    """Standard linear multi‑head attention."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__(embed_dim, num_heads, dropout)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)
        return self.out_proj(self.downstream(q, k, v, mask)[0])


# --------------------------------------------------------------------------- #
# Quantum attention
# --------------------------------------------------------------------------- #
class QuantumAttention(_BaseAttention):
    """Attention that routes projections through a lightweight quantum module."""
    class _QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev, x)
            for w, gate in enumerate(self.params):
                gate(qdev, wires=w)
            for w in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[w, w + 1])
            tqf.cnot(qdev, wires=[self.n_wires - 1, 0])
            return self.measure(qdev)

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1,
                 n_wires: int = 8, q_device: Optional[tq.QuantumDevice] = None):
        super().__init__(embed_dim, num_heads, dropout)
        self.q_layer = self._QLayer(n_wires)
        self.q_device = q_device
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        proj = self._apply_quantum(x)
        return self.out_proj(self.downstream(proj, proj, proj, mask)[0])

    def _apply_quantum(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        out = []
        for token in x.unbind(dim=1):
            qdev = self.q_device or tq.QuantumDevice(n_wires=self.q_layer.n_wires, bsz=token.size(0))
            out.append(self.q_layer(token, qdev))
        return torch.stack(out, dim=1)


# --------------------------------------------------------------------------- #
# Feed‑forward blocks
# --------------------------------------------------------------------------- #
class _BaseFFN(nn.Module):
    """Feed‑forward block – either linear or quantum."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ClassicalFFN(_BaseFFN):
    """Two‑layer MLP."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(F.relu(self.linear1(x)))


class QuantumFFN(_BaseFFN):
    """Feed‑forward realized by a quantum circuit."""
    class _QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int):
            super().__init__()
            self.n_qubits = n_qubits
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
            )
            self.params = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev, x)
            for w, gate in enumerate(self.params):
                gate(qdev, wires=w)
            return self.measure(qdev)

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1):
        super().__init__(embed_dim, ffn_dim, dropout)
        self.q_layer = self._QLayer(n_qubits)
        self.q_device = tq.QuantumDevice(n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = []
        for token in x.unbind(dim=1):
            qdev = self.q_device.copy(bsz=token.size(0))
            out.append(self.q_layer(token, qdev))
        out = torch.stack(out, dim=1)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))


# --------------------------------------------------------------------------- #
# Transformer block
# --------------------------------------------------------------------------- #
class TransformerBlock(nn.Module):
    """Hybrid transformer block that can mix classical and quantum heads."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 use_quantum_attn: bool = False,
                 use_quantum_ffn: bool = False,
                 n_wires: int = 8,
                 n_qubits: int = 8):
        super().__init__()
        self.attn = (QuantumAttention(embed_dim, num_heads, n_wires=n_wires)
                     if use_quantum_attn else ClassicalAttention(embed_dim, num_heads))
        self.ffn = (QuantumFFN(embed_dim, ffn_dim, n_qubits)
                     if use_quantum_ffn else ClassicalFFN(embed_dim, ffn_dim))
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out = self.attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


# --------------------------------------------------------------------------- #
# Positional encoding
# --------------------------------------------------------------------------- #
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000):
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
# Text classifier
# --------------------------------------------------------------------------- #
class TextClassifier(nn.Module):
    """Transformer‑based text classifier supporting quantum sub‑modules."""
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        use_quantum_attn: bool = False,
        use_quantum_ffn: bool = False,
        n_wires: int = 8,
        n_qubits: int = 8,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoding(embed_dim)
        self.transformers = nn.Sequential(
            *[
                TransformerBlock(
                    embed_dim,
                    num_heads,
                    ffn_dim,
                    use_quantum_attn=use_quantum_attn,
                    use_quantum_ffn=use_quantum_ffn,
                    n_wires=n_wires,
                    n_qubits=n_qubits,
                )
                for _ in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        x = self.transformers(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)


# --------------------------------------------------------------------------- #
# Self‑attention helper
# --------------------------------------------------------------------------- #
class UnifiedSelfAttention:
    """Convenience wrapper that exposes a classical or quantum self‑attention
    interface compatible with the original seed files."""
    def __init__(self, embed_dim: int, use_quantum: bool = False, n_wires: int = 8):
        self.use_quantum = use_quantum
        if use_quantum:
            self.attn = QuantumAttention(embed_dim, num_heads=1, n_wires=n_wires)
        else:
            self.attn = ClassicalAttention(embed_dim, num_heads=1)

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        x = torch.from_numpy(inputs).float()
        if self.use_quantum:
            out = self.attn(x, mask=None)
        else:
            out = self.attn(x)
        return out.detach().numpy()


__all__ = [
    "ClassicalAttention",
    "QuantumAttention",
    "ClassicalFFN",
    "QuantumFFN",
    "TransformerBlock",
    "PositionalEncoding",
    "TextClassifier",
    "UnifiedSelfAttention",
]
