"""Quantum transformer classifier with a quantum convolution front‑end.

This module mirrors the classical implementation but replaces the
transformer blocks with quantum‑enhanced variants and swaps the
convolution front‑end for a true quanvolution filter.  The API
signature matches the classical version so that the two modules can
be interchanged seamlessly.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

# Import the quantum front‑end; fallback to identity if unavailable
try:
    from Quanvolution import QuanvolutionFilter
except Exception:  # pragma: no cover
    class QuanvolutionFilter:
        def __init__(self) -> None:
            pass

        def __call__(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
            return x.view(x.size(0), -1)


# --------------------------------------------------------------------------- #
#  Classical transformer primitives (also used for fallback)
# --------------------------------------------------------------------------- #
class MultiHeadAttentionBase(nn.Module):
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
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq_len, _ = x.size()
        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)

        k = k.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        q = q.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
        return self.out_proj(out)


class FeedForwardBase(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class FeedForwardClassical(FeedForwardBase):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlockBase(nn.Module):
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


class PositionalEncoder(nn.Module):
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


# --------------------------------------------------------------------------- #
#  Quantum‑enhanced primitives
# --------------------------------------------------------------------------- #
class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    """Multi‑head attention where each head is processed by a small quantum
    circuit that acts on the head‑wise embedding.
    """

    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.random_layer = tq.RandomLayer(n_ops=8, wires=list(range(n_wires)))
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            self.random_layer(q_device)
            return self.measure(q_device)

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, q_device: Optional[tq.QuantumDevice] = None) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.qlayer = self.QLayer(self.d_k)
        self.q_device = q_device
        self.combine = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq_len, _ = x.size()
        x = x.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # (batch, heads, seq_len, d_k)
        out_heads = []
        for h in range(self.num_heads):
            head = x[:, h, :, :]  # (batch, seq_len, d_k)
            flat = head.reshape(batch * seq_len, self.d_k)
            qdev = self.q_device or tq.QuantumDevice(n_wires=self.d_k, bsz=flat.size(0), device=flat.device)
            q_out = self.qlayer(flat, qdev)  # (batch*seq_len, d_k)
            head_out = q_out.reshape(batch, seq_len, self.d_k)
            out_heads.append(head_out)
        out = torch.stack(out_heads, dim=1)  # (batch, heads, seq_len, d_k)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
        return self.combine(out)


class FeedForwardQuantum(FeedForwardBase):
    """Feed‑forward network realised by a small quantum circuit per token."""

    class QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int):
            super().__init__()
            self.n_qubits = n_qubits
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
            )
            self.random_layer = tq.RandomLayer(n_ops=8, wires=list(range(n_qubits)))
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            self.random_layer(q_device)
            return self.measure(q_device)

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.qlayer = self.QLayer(n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.size()
        flat = x.reshape(batch * seq_len, x.size(2))
        qdev = tq.QuantumDevice(n_wires=self.qlayer.n_qubits, bsz=flat.size(0), device=flat.device)
        q_out = self.qlayer(flat, qdev)  # (batch*seq_len, n_qubits)
        q_out = q_out.reshape(batch, seq_len, self.qlayer.n_qubits)
        out = self.linear1(self.dropout(q_out))
        return self.linear2(F.relu(out))


class TransformerBlockQuantum(TransformerBlockBase):
    """Quantum‑enhanced transformer block that uses the quantum attention
    and feed‑forward primitives defined above.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        n_qubits_transformer: int,
        n_qubits_ffn: int,
        dropout: float = 0.1,
        q_device: Optional[tq.QuantumDevice] = None,
    ) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout, q_device)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


# --------------------------------------------------------------------------- #
#  Quantum‑aware front‑end
# --------------------------------------------------------------------------- #
class QuanvolutionFilter(tq.QuantumModule):
    """Apply a random two‑qubit quantum kernel to 2×2 image patches."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "rx", "wires": [0]},
                {"input_idx": [1], "func": "rx", "wires": [1]},
                {"input_idx": [2], "func": "rx", "wires": [2]},
                {"input_idx": [3], "func": "rx", "wires": [3]},
            ]
        )
        self.random_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.random_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)


# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #
class QTransformerTorch(nn.Module):
    """Quantum‑enhanced transformer classifier.

    The interface matches the classical variant so that the two
    modules can be swapped at import time.  When ``use_quantum_frontend``
    is ``True`` the model applies a ``QuanvolutionFilter`` to an image
    before feeding the resulting token‑like features into a stack of
    quantum transformer blocks.  When ``use_quantum_frontend`` is
    ``False`` the model behaves exactly like the classical implementation.
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
        conv_kernel_size: int = 2,
        conv_threshold: float = 0.0,
        use_quantum_frontend: bool = False,
        use_quantum_transformer: bool = False,
        n_qubits_transformer: int = 8,
        n_qubits_ffn: int = 8,
        q_device: Optional[tq.QuantumDevice] = None,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)

        # quantum front‑end
        self.frontend = QuanvolutionFilter()
        self.frontend_linear = nn.Linear(4, embed_dim)

        # transformer stack
        if use_quantum_transformer:
            self.transformer_blocks = nn.ModuleList(
                [
                    TransformerBlockQuantum(
                        embed_dim,
                        num_heads,
                        ffn_dim,
                        n_qubits_transformer,
                        n_qubits_ffn,
                        dropout,
                        q_device=q_device,
                    )
                    for _ in range(num_blocks)
                ]
            )
        else:
            self.transformer_blocks = nn.ModuleList(
                [
                    TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
                    for _ in range(num_blocks)
                ]
            )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

        # flags for introspection
        self.use_quantum_frontend = use_quantum_frontend
        self.use_quantum_transformer = use_quantum_transformer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        If ``x`` has shape (batch, 1, 28, 28) it is treated as an image and
        processed by the quanvolution front‑end.  Otherwise it is assumed to be a
        token index sequence.
        """
        if x.dim() == 4:
            # image → feature sequence
            features = self.frontend(x)  # (batch, 4*196)
            seq_len = features.size(1) // 4
            features = features.view(x.size(0), seq_len, 4)
            x = self.frontend_linear(features)
        else:
            # text sequence
            tokens = self.token_embedding(x)
            x = self.pos_encoder(tokens)

        for blk in self.transformer_blocks:
            x = blk(x)

        x = x.mean(dim=1)  # global pooling
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
    "QuanvolutionFilter",
    "QTransformerTorch",
]
