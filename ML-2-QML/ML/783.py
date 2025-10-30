"""Hybrid transformer with optional quantum attention and feed‑forward modules.

The implementation keeps the original classical API but exposes a new
`HybridTransformer` class that can be instantiated with a quantum
attention depth and a feed‑forward quantum block.  The design follows
the same layer‑wise structure as the seed, but the quantum parts are
parameter‑shared across heads and use a lightweight
parameter‑efficient variational circuit.  This allows quick
experiments without a full TorchQuantum stack while still
demonstrating quantum‑enhanced attention.
"""

from __future__ import annotations

import math
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

# --------------------------------------------------------------------------- #
# 1. Core building blocks
# --------------------------------------------------------------------------- #

class MultiHeadAttentionBase(nn.Module):
    """Base class for all attention variants."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.d_k = embed_dim // num_heads

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError


class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Purely classical multi‑head attention."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:  # type: ignore[override]
        batch, seq, _ = x.shape
        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)
        # split heads
        k = k.view(batch, seq, self.num_heads, self.d_k)
        q = q.view(batch, seq, self.num_heads, self.d_k)
        v = v.view(batch, seq, self.num_heads, self.d_k)
        # transpose to (batch, heads, seq, d_k)
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # scaled dot‑product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_out = torch.matmul(attn_weights, v)
        # concat heads
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq, self.embed_dim)
        return self.out_proj(attn_out)


class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    """Quantum‑based attention that maps projections through a variational circuit."""
    class QLayer(tq.QuantumModule):
        def __init__(self, d_k: int, depth: int):
            super().__init__()
            self.d_k = d_k
            self.depth = depth
            # Encode classical input into quantum state via RX gates
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(d_k)]
            )
            # Trainable rotation parameters per depth
            self.params = nn.ParameterList(
                [nn.Parameter(torch.randn(d_k)) for _ in range(depth)]
            )

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            # x shape: (batch, d_k)
            self.encoder(q_device, x)
            for p in self.params:
                for i in range(self.d_k):
                    tq.RX(q_device, wires=i)(p[i])
            # simple entanglement via CNOT chain
            for _ in range(self.depth):
                for i in range(self.d_k - 1):
                    tqf.cnot(q_device, wires=[i, i + 1])
            return tq.MeasureAll(tq.PauliZ)(q_device)

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1,
                 quantum_depth: int = 2, share_weights: bool = True) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.quantum_depth = quantum_depth
        self.share_weights = share_weights
        if share_weights:
            self.q_layer = self.QLayer(self.d_k, quantum_depth)
            self.q_layers = [self.q_layer] * num_heads
        else:
            self.q_layers = [self.QLayer(self.d_k, quantum_depth) for _ in range(num_heads)]

    def _apply_q_layer(self, x: torch.Tensor, q_layer: tq.QuantumModule) -> torch.Tensor:
        # x shape: (batch, seq, d_k)
        batch, seq, d_k = x.shape
        out = []
        for i in range(seq):
            token = x[:, i, :]
            qdev = tq.QuantumDevice(n_wires=d_k, bsz=batch, device=token.device)
            out.append(q_layer(token, qdev))
        return torch.stack(out, dim=1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq, _ = x.shape
        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)
        # split heads
        k = k.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        q = q.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        # apply quantum layer per head
        attn_out = []
        for head_idx in range(self.num_heads):
            k_head = k[:, head_idx, :, :]
            q_head = q[:, head_idx, :, :]
            v_head = v[:, head_idx, :, :]
            k_head_q = self._apply_q_layer(k_head, self.q_layers[head_idx])
            q_head_q = self._apply_q_layer(q_head, self.q_layers[head_idx])
            v_head_q = self._apply_q_layer(v_head, self.q_layers[head_idx])
            # scaled dot‑product attention
            scores = torch.matmul(q_head_q, k_head_q.transpose(-2, -1)) / math.sqrt(self.d_k)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attn_out.append(torch.matmul(attn_weights, v_head_q))
        # concatenate heads
        attn_out = torch.stack(attn_out, dim=1).transpose(1, 2).contiguous()
        attn_out = attn_out.view(batch, seq, self.embed_dim)
        return self.out_proj(attn_out)


class FeedForwardBase(nn.Module):
    """Base class for feed‑forward layers."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class FeedForwardClassical(FeedForwardBase):
    """Two‑layer perceptron feed‑forward network."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.linear2 = nn.Linear(ffn_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class FeedForwardQuantum(FeedForwardBase):
    """Feed‑forward network realised by a quantum module."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            # Encode classical input into quantum state
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            # Trainable rotation parameters
            self.params = nn.ParameterList(
                [nn.Parameter(torch.randn(n_wires)) for _ in range(1)]
            )

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for p in self.params:
                for i in range(self.n_wires):
                    tq.RX(q_device, wires=i)(p[i])
            return tq.MeasureAll(tq.PauliZ)(q_device)

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.n_qubits = n_qubits
        self.q_layer = self.QLayer(n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim, bias=False)
        self.linear2 = nn.Linear(ffn_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        out = []
        for i in range(seq):
            token = x[:, i, :]
            qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=batch, device=token.device)
            out.append(self.q_layer(token, qdev))
        out = torch.stack(out, dim=1)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))


class TransformerBlockBase(nn.Module):
    """Base transformer block containing attention and feed‑forward parts."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TransformerBlockClassical(TransformerBlockBase):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class TransformerBlockQuantum(TransformerBlockBase):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 quantum_depth: int = 2, share_weights: bool = True,
                 n_qubits_ffn: int = 0, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout,
                                               quantum_depth=quantum_depth,
                                               share_weights=share_weights)
        if n_qubits_ffn > 0:
            self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, dropout)
        else:
            self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

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
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class HybridTransformer(nn.Module):
    """Transformer-based text classifier supporting optional quantum submodules."""
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        quantum_depth: int = 0,
        share_weights: bool = True,
        n_qubits_ffn: int = 0,
        use_quantum_ffn: bool = False,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoder(embed_dim)
        if quantum_depth > 0:
            blocks: List[nn.Module] = []
            for _ in range(num_blocks):
                block = TransformerBlockQuantum(
                    embed_dim,
                    num_heads,
                    ffn_dim,
                    quantum_depth=quantum_depth,
                    share_weights=share_weights,
                    n_qubits_ffn=n_qubits_ffn,
                    dropout=dropout,
                )
                blocks.append(block)
            self.transformer = nn.Sequential(*blocks)
        else:
            self.transformer = nn.Sequential(
                *[TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
                  for _ in range(num_blocks)]
            )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        x = self.pos_encoding(x)
        x = self.transformer(x)
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
    "TransformerBlockClassical",
    "TransformerBlockQuantum",
    "PositionalEncoder",
    "HybridTransformer",
]
