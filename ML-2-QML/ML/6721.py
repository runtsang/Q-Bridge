"""Hybrid transformer with optional quantum submodules.

This module keeps the original classical API but adds a lightweight quantum hook.
The `quantum_enabled` flag controls whether the attention and feed‑forward layers
use their quantum counterparts. The class can be swapped into the original
pipeline without changes in downstream code.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional quantum backend
try:
    import torchquantum as tq
    import torchquantum.functional as tqf
except Exception:  # pragma: no cover
    tq = None  # type: ignore[assignment]
    tqf = None  # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# Base building blocks – identical to the seed implementation
# --------------------------------------------------------------------------- #
class MultiHeadAttentionBase(nn.Module):
    """Base class for multi‑head attention."""
    def __init__(self, embed_dim: int, num_heads: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.d_k = embed_dim // num_heads


class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Standard multi‑head attention using nn.MultiheadAttention."""
    def __init__(self, embed_dim: int, num_heads: int,
                 dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_output, _ = self.attn(x, x, x,
                                   key_padding_mask=mask)
        return attn_output


# Quantum attention – falls back to classical if torchquantum unavailable
if tq is not None:
    class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
        """Multi‑head attention where each head is processed by a small quantum circuit."""
        class QLayer(tq.QuantumModule):
            def __init__(self, n_wires: int):
                super().__init__()
                self.n_wires = n_wires
                self.encoder = tq.GeneralEncoder(
                    [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
                )
                self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True)
                                             for _ in range(n_wires)])
                self.measure = tq.MeasureAll(tq.PauliZ)

            def forward(self, x: torch.Tensor,
                        q_device: tq.QuantumDevice) -> torch.Tensor:
                self.encoder(q_device, x)
                for wire, gate in enumerate(self.params):
                    gate(q_device, wires=wire)
                return self.measure(q_device)

        def __init__(self, embed_dim: int, num_heads: int,
                     dropout: float = 0.1,
                     q_device: Optional[tq.QuantumDevice] = None) -> None:
            super().__init__(embed_dim, num_heads, dropout)
            self.q_layer = self.QLayer(self.num_heads * 2)
            self.q_device = q_device
            self.combine = nn.Linear(embed_dim, embed_dim)

        def forward(self, x: torch.Tensor,
                    mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            batch, seq, _ = x.shape
            # Split heads
            x_heads = x.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
            out_heads = []
            for head in x_heads.unbind(dim=1):
                flat = head.reshape(-1, self.d_k)
                qdev = self.q_device or tq.QuantumDevice(
                    n_wires=self.q_layer.n_wires,
                    bsz=flat.size(0),
                    device=flat.device
                )
                qout = self.q_layer(flat, qdev)
                out_heads.append(qout)
            out = torch.stack(out_heads, dim=1)  # (batch, seq, num_heads, d_k)
            out = out.transpose(1, 2).contiguous().view(batch, seq, self.embed_dim)
            return self.combine(out)
else:
    class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
        """Fallback that raises an error if torchquantum is missing."""
        def __init__(self, *args, **kwargs):
            raise ImportError("torchquantum is required for MultiHeadAttentionQuantum")


class FeedForwardBase(nn.Module):
    """Base feed‑forward network."""
    def __init__(self, embed_dim: int, ffn_dim: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)


class FeedForwardClassical(FeedForwardBase):
    """Two‑layer perceptron feed‑forward."""
    def __init__(self, embed_dim: int, ffn_dim: int,
                 dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# Quantum feed‑forward – falls back if torchquantum unavailable
if tq is not None:
    class FeedForwardQuantum(FeedForwardBase):
        """Feed‑forward realized by a quantum circuit."""
        class QLayer(tq.QuantumModule):
            def __init__(self, n_qubits: int):
                super().__init__()
                self.n_qubits = n_qubits
                self.encoder = tq.GeneralEncoder(
                    [{"input_idx": [i], "func": "ry", "wires": [i]}
                     for i in range(n_qubits)]
                )
                self.params = nn.ModuleList([tq.RZ(has_params=True, trainable=True)
                                             for _ in range(n_qubits)])
                self.measure = tq.MeasureAll(tq.PauliZ)

            def forward(self, x: torch.Tensor,
                        q_device: tq.QuantumDevice) -> torch.Tensor:
                self.encoder(q_device, x)
                for wire, gate in enumerate(self.params):
                    gate(q_device, wires=wire)
                return self.measure(q_device)

        def __init__(self, embed_dim: int, ffn_dim: int,
                     n_qubits: int, dropout: float = 0.1) -> None:
            super().__init__(embed_dim, ffn_dim, dropout)
            self.q_layer = self.QLayer(n_qubits)
            self.linear1 = nn.Linear(n_qubits, ffn_dim)
            self.linear2 = nn.Linear(ffn_dim, embed_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            outputs = []
            for token in x.unbind(dim=1):
                qdev = tq.QuantumDevice(
                    n_wires=self.q_layer.n_qubits,
                    bsz=token.size(0),
                    device=token.device
                )
                out = self.q_layer(token, qdev)
                outputs.append(out)
            out = torch.stack(outputs, dim=1)
            out = self.linear1(self.dropout(out))
            return self.linear2(F.relu(out))
else:
    class FeedForwardQuantum(FeedForwardBase):
        """Fallback that raises an error if torchquantum is missing."""
        def __init__(self, *args, **kwargs):
            raise ImportError("torchquantum is required for FeedForwardQuantum")


class TransformerBlockBase(nn.Module):
    """Base transformer block."""
    def __init__(self, embed_dim: int, num_heads: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)


class TransformerBlockClassical(TransformerBlockBase):
    def __init__(self, embed_dim: int, num_heads: int,
                 ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class TransformerBlockQuantum(TransformerBlockBase):
    def __init__(self, embed_dim: int, num_heads: int,
                 ffn_dim: int, n_qubits_transformer: int,
                 n_qubits_ffn: int,
                 q_device: Optional[tq.QuantumDevice] = None,
                 dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads,
                                              dropout, q_device=q_device)
        if n_qubits_ffn > 0:
            self.ffn = FeedForwardQuantum(embed_dim, ffn_dim,
                                          n_qubits_ffn, dropout)
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
    """Transformer-based text classifier supporting optional quantum submodules."""
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 quantum_enabled: bool = False,
                 n_qubits_transformer: int = 0,
                 n_qubits_ffn: int = 0,
                 q_device: Optional[tq.QuantumDevice] = None) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        if quantum_enabled:
            if tq is None:
                raise ImportError("torchquantum is required for quantum-enabled model")
            self.transformers = nn.Sequential(
                *[TransformerBlockQuantum(embed_dim, num_heads, ffn_dim,
                                          n_qubits_transformer, n_qubits_ffn,
                                          q_device=q_device, dropout=dropout)
                  for _ in range(num_blocks)]
            )
        else:
            self.transformers = nn.Sequential(
                *[TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
                  for _ in range(num_blocks)]
            )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim,
                                    num_classes if num_classes > 2 else 1)

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
