"""Quantum‑hybrid transformer mirroring the classical API.

The module implements the same class name, ``HybridTransformer``,
but replaces the attention and feed‑forward sub‑modules with
quantum‑enabled counterparts based on TorchQuantum.  The
synthetic regression utilities are also adapted to work with
complex quantum states.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np
from torch.utils.data import Dataset


# --------------------------------------------------------------------------- #
#  Utility: synthetic superposition data (quantum version)
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Create quantum states of the form
    cos(theta)|0...0⟩ + e^{i phi} sin(theta)|1...1⟩
    and a target function sin(2*theta)*cos(phi).
    """
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels


class RegressionDataset(Dataset):
    """Dataset that yields complex quantum state vectors."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


# --------------------------------------------------------------------------- #
#  Quantum‑enabled transformer components
# --------------------------------------------------------------------------- #
class MultiHeadAttentionBase(nn.Module):
    """Shared logic for attention layers."""
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.1,
                 mask: Optional[torch.Tensor] = None,
                 use_bias: bool = False) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError(
                f"Embedding dimension ({embed_dim}) must be divisible by number of heads ({num_heads})"
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

    def attention(self,
                  query: torch.Tensor,
                  key: torch.Tensor,
                  value: torch.Tensor,
                  mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        return torch.matmul(scores, value), scores

    def downstream(self,
                   query: torch.Tensor,
                   key: torch.Tensor,
                   value: torch.Tensor,
                   batch_size: int,
                   mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = self.separate_heads(query)
        k = self.separate_heads(key)
        v = self.separate_heads(value)
        out, _ = self.attention(q, k, v, mask)
        return out.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)


class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Standard multi‑head attention using nn.MultiheadAttention."""
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.1,
                 mask: Optional[torch.Tensor] = None,
                 use_bias: bool = False) -> None:
        super().__init__(embed_dim, num_heads, dropout, mask, use_bias)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_out


class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    """Quantum‑enhanced attention that projects queries/keys/values through a small
    parameter‑ised quantum circuit before the classical attention operation."""
    class QLayer(tq.QuantumModule):
        """Fixed‑depth circuit that acts on an 8‑wire subsystem."""
        def __init__(self) -> None:
            super().__init__()
            self.n_wires = 8
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [i], "func": "rx", "wires": [i]}
                    for i in range(self.n_wires)
                ]
            )
            self.parameters = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(self.n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for wire, gate in enumerate(self.parameters):
                gate(q_device, wires=wire)
            # entangle all wires in a ring
            for wire in range(self.n_wires - 1):
                tqf.cnot(q_device, wires=[wire, wire + 1])
            tqf.cnot(q_device, wires=[self.n_wires - 1, 0])
            return self.measure(q_device)

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.1,
                 mask: Optional[torch.Tensor] = None,
                 use_bias: bool = False,
                 q_device: Optional[tq.QuantumDevice] = None) -> None:
        super().__init__(embed_dim, num_heads, dropout, mask, use_bias)
        self.q_layer = self.QLayer()
        self.q_device = q_device
        self.combine_heads = nn.Linear(embed_dim, embed_dim, bias=use_bias)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, _, embed_dim = x.size()
        if embed_dim!= self.embed_dim:
            raise ValueError(
                f"Input embedding ({embed_dim}) does not match layer embedding size ({self.embed_dim})"
            )
        # Apply the quantum circuit to each projection
        k = self._apply_quantum_heads(x)
        q = self._apply_quantum_heads(x)
        v = self._apply_quantum_heads(x)
        out = self.downstream(q, k, v, batch_size, mask)
        return self.combine_heads(out)

    def _apply_quantum_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the quantum circuit to each head of every token."""
        projections = []
        for token in x.unbind(dim=1):
            token = token.view(token.size(0), self.num_heads, -1)
            head_outs = []
            for head in token.unbind(dim=1):
                qdev = self.q_device or tq.QuantumDevice(
                    n_wires=self.q_layer.n_wires, bsz=head.size(0), device=head.device
                )
                head_outs.append(self.q_layer(head, qdev))
            projections.append(torch.stack(head_outs, dim=1))
        return torch.stack(projections, dim=1)


class FeedForwardBase(nn.Module):
    """Shared interface for feed‑forward layers."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)


class FeedForwardClassical(FeedForwardBase):
    """Two‑layer MLP feed‑forward network."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class FeedForwardQuantum(FeedForwardBase):
    """Feed‑forward realized by a small quantum circuit followed by classical layers."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int) -> None:
            super().__init__()
            self.n_wires = n_qubits
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [idx], "func": "rx", "wires": [idx]}
                    for idx in range(n_qubits)
                ]
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

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.q_layer = self.QLayer(n_qubits)
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = []
        for token in x.unbind(dim=1):
            qdev = self.q_device.copy(bsz=token.size(0), device=token.device)
            outs.append(self.q_layer(token, qdev))
        out = torch.stack(outs, dim=1)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))


class TransformerBlockBase(nn.Module):
    """Base transformer block with optional quantum sub‑modules."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class TransformerBlockClassical(TransformerBlockBase):
    """Classical transformer block."""
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class TransformerBlockQuantum(TransformerBlockBase):
    """Quantum‑enabled transformer block."""
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 n_qubits_transformer: int,
                 n_qubits_ffn: int,
                 n_qlayers: int,
                 q_device: Optional[tq.QuantumDevice] = None,
                 dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout,
                                              q_device=q_device)
        self.ffn = (FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, dropout)
                    if n_qubits_ffn > 0 else FeedForwardClassical(embed_dim, ffn_dim, dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) *
                             (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


# --------------------------------------------------------------------------- #
#  HybridTransformer: unified API for classical, quantum, and regression
# --------------------------------------------------------------------------- #
class HybridTransformer(tq.QuantumModule):
    """
    Unified transformer that can operate in classical mode (default) or
    with quantum‑enhanced sub‑modules.  The public API is identical to the
    classical implementation above, which simplifies switching between
    back‑ends during experimentation.
    """
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 use_quantum: bool = False,
                 n_qubits_transformer: int = 0,
                 n_qubits_ffn: int = 0,
                 n_qlayers: int = 1) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)

        # Instantiate the appropriate block type
        if use_quantum and n_qubits_transformer > 0:
            qdev = tq.QuantumDevice(n_wires=max(n_qubits_transformer, n_qubits_ffn))
            blocks = [
                TransformerBlockQuantum(
                    embed_dim,
                    num_heads,
                    ffn_dim,
                    n_qubits_transformer,
                    n_qubits_ffn,
                    n_qlayers,
                    q_device=qdev,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        else:
            blocks = [
                TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
                for _ in range(num_blocks)
            ]
        self.transformers = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout)
        self.num_classes = num_classes
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 1 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        x = self.transformers(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)


__all__ = [
    "HybridTransformer",
    "generate_superposition_data",
    "RegressionDataset",
]
