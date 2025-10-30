"""Hybrid transformer that can toggle quantum modules per block.

This QML module implements the same API as the classical version but
provides lightweight quantum attention and feed‑forward circuits
using TorchQuantum.  The circuits are fully parameter‑shared across
heads and tokens to keep simulation cost low.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class MultiHeadAttentionBase(nn.Module):
    """Base class for multi‑head attention, shared by classical and quantum variants."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.attn_weights: None | torch.Tensor = None

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape to (batch, heads, seq_len, d_k)."""
        batch_size = x.size(0)
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

    def attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: None | torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Scaled dot‑product attention."""
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        return torch.matmul(scores, value), scores

    def downstream(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        batch_size: int,
        mask: None | torch.Tensor = None,
    ) -> torch.Tensor:
        """Apply attention heads and merge back."""
        q = self.separate_heads(query)
        k = self.separate_heads(key)
        v = self.separate_heads(value)
        out, self.attn_weights = self.attention(q, k, v, mask)
        return out.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)


class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Pure‑Python implementation of multi‑head attention."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.combine_heads = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: None | torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)
        out = self.downstream(q, k, v, batch_size, mask)
        return self.combine_heads(out)


class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    """Quantum‑enhanced attention that re‑uses a variational circuit per head."""

    class _QLayer(tq.QuantumModule):
        """Variational circuit that maps a classical token to a quantum state."""

        def __init__(self, n_wires: int = 8):
            super().__init__()
            self.n_wires = n_wires
            # Encode each qubit with an RX gate controlled by the token embedding
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [idx], "func": "rx", "wires": [idx]} for idx in range(n_wires)]
            )
            # Trainable rotation parameters
            self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, token: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            # token shape: (batch, n_wires)
            self.encoder(q_device, token)
            for w, gate in enumerate(self.params):
                gate(q_device, wires=w)
            return self.measure(q_device)

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, q_device: None | tq.QuantumDevice = None):
        super().__init__(embed_dim, num_heads, dropout)
        self.q_layer = self._QLayer()
        self.q_device = q_device or tq.QuantumDevice(n_wires=self.q_layer.n_wires)
        self.combine_heads = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: None | torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        # Project each token to quantum heads
        proj = self._apply_heads(x)
        out = self.downstream(proj, proj, proj, batch_size, mask)
        return self.combine_heads(out)

    def _apply_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the same variational circuit to every token in every head."""
        outputs = []
        for token in x.unbind(dim=1):  # iterate over sequence
            # token shape: (batch, embed_dim)
            token = token.view(token.size(0), self.num_heads, -1)  # split into heads
            head_outs = []
            for head in token.unbind(dim=1):
                # Each head has shape (batch, d_k)
                qdev = self.q_device.copy(bsz=head.size(0), device=head.device)
                head_outs.append(self.q_layer(head, qdev))
            outputs.append(torch.stack(head_outs, dim=1))
        return torch.stack(outputs, dim=1)


class FeedForwardBase(nn.Module):
    """Base class for feed‑forward layers."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)


class FeedForwardClassical(nn.Module):
    """Standard two‑layer MLP."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class FeedForwardQuantum(FeedForwardBase):
    """Quantum‑based feed‑forward network using a small variational circuit."""

    class _QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int):
            super().__init__()
            self.n_qubits = n_qubits
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [idx], "func": "rx", "wires": [idx]} for idx in range(n_qubits)]
            )
            self.params = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, token: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            # token shape: (batch, n_qubits)
            self.encoder(q_device, token)
            for w, gate in enumerate(self.params):
                gate(q_device, wires=w)
            return self.measure(q_device)

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.q_layer = self._QLayer(n_qubits)
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for token in x.unbind(dim=1):  # iterate over sequence
            qdev = self.q_device.copy(bsz=token.size(0), device=token.device)
            outputs.append(self.q_layer(token, qdev))
        out = torch.stack(outputs, dim=1)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))


class TransformerBlockBase(nn.Module):
    """Base transformer block containing attention and feed‑forward parts."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface only
        raise NotImplementedError


class TransformerBlockHybrid(TransformerBlockBase):
    """Transformer block that can switch between classical and quantum sub‑modules."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        use_quantum_attn: bool = False,
        use_quantum_ffn: bool = False,
    ) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = (
            MultiHeadAttentionQuantum(embed_dim, num_heads, dropout)
            if use_quantum_attn
            else MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        )
        self.ffn = (
            FeedForwardQuantum(embed_dim, ffn_dim, n_qubits=embed_dim // num_heads, dropout=dropout)
            if use_quantum_ffn
            else FeedForwardClassical(embed_dim, ffn_dim, dropout)
        )

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
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TextClassifier(nn.Module):
    """Transformer‑based text classifier supporting hybrid quantum blocks."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        use_quantum_attn: Iterable[bool] | None = None,
        use_quantum_ffn: Iterable[bool] | None = None,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)

        if use_quantum_attn is None:
            use_quantum_attn = [False] * num_blocks
        if use_quantum_ffn is None:
            use_quantum_ffn = [False] * num_blocks

        blocks = []
        for i in range(num_blocks):
            blocks.append(
                TransformerBlockHybrid(
                    embed_dim,
                    num_heads,
                    ffn_dim,
                    dropout,
                    use_quantum_attn=use_quantum_attn[i],
                    use_quantum_ffn=use_quantum_ffn[i],
                )
            )
        self.transformers = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(
            embed_dim, num_classes if num_classes > 2 else 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        x = self.transformers(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)


__all__ = [
    "MultiHeadAttentionBase",
    "MultiHeadAttentionClassical",
    "MultiHeadAttentionQuantum",
    "FeedForwardBase",
    "FeedForwardClassical",
    "FeedForwardQuantum",
    "TransformerBlockBase",
    "TransformerBlockHybrid",
    "PositionalEncoder",
    "TextClassifier",
]
