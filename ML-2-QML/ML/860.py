"""Hybrid classical‑quantum transformer for text classification."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq


class MultiHeadAttentionBase(nn.Module):
    """Base class providing common utilities for classical and quantum heads."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape to (batch, heads, seq, d_k)."""
        batch, seq, _ = x.shape
        return x.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse of separate_heads."""
        return x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.embed_dim)

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                  mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Scaled dot‑product attention."""
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1), -1e9)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.bmm(self.dropout(attn_weights), v)
        return attn_output, attn_weights

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Standard multi‑head attention implemented with nn.MultiheadAttention."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1,
                 bias: bool = True) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout,
                                          batch_first=True)

    def forward(self, *args, **kwargs) -> torch.Tensor:  # type: ignore[override]
        x = self.proj(*args, **kwargs)
        attn_out, _ = self.attn(x, x, x)
        return self.combine_heads(attn_out)


class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    """Quantum‑augmented attention that shares a projection matrix across heads."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1,
                 n_qubits: int = 8, n_layers: int = 1, bias: bool = False) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.shared_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.n_qubits = min(self.d_k, n_qubits)
        self.qc_layers = nn.ModuleList([self._build_qc() for _ in range(n_layers)])
        self.linear_to_dk = nn.Linear(self.n_qubits, self.d_k)

    def _build_qc(self) -> nn.Module:
        """Return a simple parameterised circuit for a single layer."""
        class _QC(nn.Module):
            def __init__(self, n_wires: int):
                super().__init__()
                self.n_wires = n_wires
                self.encoder = tq.GeneralEncoder(
                    [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
                )
                self.parameters = nn.ParameterList(
                    [nn.Parameter(torch.randn(1)) for _ in range(n_wires)]
                )
                self.measure = tq.MeasureAll(tq.PauliZ)

            def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
                self.encoder(q_device, x)
                for i, param in enumerate(self.parameters):
                    tq.RX(param, wires=i)(q_device)
                return self.measure(q_device)

        return _QC(self.n_qubits)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq, _ = x.shape
        proj = self.shared_proj(x)
        heads = self.separate_heads(proj)
        processed_heads = []
        for h in range(self.num_heads):
            head = heads[:, h, :, :]  # (batch, seq, d_k)
            flat = head.reshape(-1, self.d_k)
            flat_q = flat[:, :self.n_qubits]
            qdev = tq.QuantumDevice(n_wires=self.n_qubits,
                                    bsz=flat_q.shape[0],
                                    device=flat_q.device)
            q_out = flat_q
            for qc in self.qc_layers:
                q_out = qc(q_out, qdev)
            q_out = self.linear_to_dk(q_out)
            q_out = q_out.reshape(batch, seq, self.d_k)
            processed_heads.append(q_out)
        heads_out = torch.stack(processed_heads, dim=1)
        combined = self.combine_heads(heads_out)
        attn_out, _ = self.attention(combined, combined, combined, mask)
        return attn_out


class FeedForwardBase(nn.Module):
    """Base class for feed‑forward networks."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class FeedForwardClassical(FeedForwardBase):
    """Two‑layer perceptron feed‑forward network."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class FeedForwardQuantum(FeedForwardBase):
    """Feed‑forward network realised by a quantum module."""

    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.parameters = nn.ParameterList(
                [nn.Parameter(torch.randn(1)) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for i, param in enumerate(self.parameters):
                tq.RX(param, wires=i)(q_device)
            return self.measure(q_device)

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1):
        super().__init__(embed_dim, ffn_dim, dropout)
        self.n_qubits = min(embed_dim, n_qubits)
        self.q_layer = self.QLayer(self.n_qubits)
        self.q_device = tq.QuantumDevice(n_wires=self.n_qubits)
        self.linear1 = nn.Linear(self.n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        flat = x.reshape(-1, x.shape[-1])
        flat_q = flat[:, :self.n_qubits]
        qdev = self.q_device.copy(bsz=flat_q.shape[0], device=flat_q.device)
        q_out = self.q_layer(flat_q, qdev)
        q_out = self.linear1(q_out)
        q_out = F.relu(q_out)
        q_out = self.linear2(q_out)
        return q_out.reshape(batch, seq, -1)


class TransformerBlockBase(nn.Module):
    """Base transformer block containing attention and feed‑forward parts."""

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
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
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
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 n_qubits_transformer: int, n_qubits_ffn: int, n_qlayers: int,
                 q_device: Optional[tq.QuantumDevice] = None,
                 dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout,
                                              n_qubits=n_qubits_transformer,
                                              n_layers=n_qlayers)
        if n_qubits_ffn > 0:
            self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits=n_qubits_ffn,
                                          dropout=dropout)
        else:
            self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding with optional learnable bias."""

    def __init__(self, embed_dim: int, max_len: int = 5000, learnable_bias: bool = False):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) *
                             (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
        if learnable_bias:
            self.bias = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x + self.pe[:, :x.size(1)]
        if self.bias is not None:
            out = out + self.bias
        return out


class TextClassifier(nn.Module):
    """Transformer‑based text classifier supporting quantum submodules."""

    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 n_qubits_transformer: int = 0,
                 n_qubits_ffn: int = 0,
                 n_qlayers: int = 1,
                 q_device: Optional[tq.QuantumDevice] = None,
                 learnable_pos_bias: bool = False) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim, learnable_bias=learnable_pos_bias)
        if n_qubits_transformer > 0:
            q_device = q_device or tq.QuantumDevice(
                n_wires=max(n_qubits_transformer, n_qubits_ffn))
            blocks = [
                TransformerBlockQuantum(embed_dim,
                                       num_heads,
                                       ffn_dim,
                                       n_qubits_transformer,
                                       n_qubits_ffn,
                                       n_qlayers,
                                       q_device=q_device,
                                       dropout=dropout)
                for _ in range(num_blocks)
            ]
        else:
            blocks = [
                TransformerBlockClassical(embed_dim,
                                         num_heads,
                                         ffn_dim,
                                         dropout=dropout)
                for _ in range(num_blocks)
            ]
        self.transformers = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

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
    "TransformerBlockClassical",
    "TransformerBlockQuantum",
    "PositionalEncoder",
    "TextClassifier",
]
