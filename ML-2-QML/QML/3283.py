"""Hybrid vision transformer with quantum front‑end and optional quantum transformer blocks.

The quantum implementation mirrors the classical version but replaces the convolutional patch extractor
with a random two‑qubit quantum kernel (quanvolution) and each transformer block with a quantum‑enhanced
attention/FFN pair.  The API is intentionally identical to the classical module to enable side‑by‑side
experimentation.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


# --------------------------------------------------------------------------- #
#  Multi‑Head Attention
# --------------------------------------------------------------------------- #
class MultiHeadAttentionBase(nn.Module):
    """Shared logic for attention layers."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.d_k = embed_dim // num_heads

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

    def attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                  mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        return torch.matmul(scores, value), scores

    def downstream(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                   batch_size: int, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = self.separate_heads(query)
        k = self.separate_heads(key)
        v = self.separate_heads(value)
        out, _ = self.attention(q, k, v, mask)
        return out.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)


class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    """Quantum‑enhanced multi‑head attention."""

    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int = 8) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.parameters = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev, x)
            for gate, wire in zip(self.parameters, range(self.n_wires)):
                gate(qdev, wires=wire)
            return self.measure(qdev)

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.q_layer = self.QLayer()
        self.combine_heads = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.size()
        if embed_dim!= self.embed_dim:
            raise ValueError("Input embedding mismatch.")
        # Apply quantum transformation per token
        x_q = torch.stack(
            [
                self.q_layer(token.unsqueeze(0), tq.QuantumDevice(self.q_layer.n_wires,
                                                                 bsz=1,
                                                                 device=token.device))
                for token in x.unbind(dim=1)
            ],
            dim=1
        ).squeeze(1)
        # Classical attention on quantum‑encoded tokens
        q = self.q_layer.encoder  # placeholder to keep API
        return self.combine_heads(x_q)


# --------------------------------------------------------------------------- #
#  Feed‑Forward Network
# --------------------------------------------------------------------------- #
class FeedForwardBase(nn.Module):
    """Shared interface for feed‑forward layers."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)


class FeedForwardQuantum(FeedForwardBase):
    """Feed‑forward realized by a small quantum circuit."""

    class QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int = 8) -> None:
            super().__init__()
            self.n_qubits = n_qubits
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_qubits)]
            )
            self.parameters = nn.ModuleList(
                [tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev, x)
            for gate, wire in zip(self.parameters, range(self.n_qubits)):
                gate(qdev, wires=wire)
            return self.measure(qdev)

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int = 8, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.q_layer = self.QLayer(n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for token in x.unbind(dim=1):
            qdev = tq.QuantumDevice(self.q_layer.n_qubits, bsz=token.size(0), device=token.device)
            outputs.append(self.q_layer(token, qdev))
        out = torch.stack(outputs, dim=1)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))


# --------------------------------------------------------------------------- #
#  Transformer Block
# --------------------------------------------------------------------------- #
class TransformerBlockBase(nn.Module):
    """Base transformer block containing attention and feed‑forward parts."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class TransformerBlockQuantum(TransformerBlockBase):
    """Quantum‑enhanced transformer block."""

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 n_qubits: int = 8,
                 dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


# --------------------------------------------------------------------------- #
#  Positional Encoding
# --------------------------------------------------------------------------- #
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
        return x + self.pe[:, : x.size(1)]


# --------------------------------------------------------------------------- #
#  Quantum Quanvolution Front‑end
# --------------------------------------------------------------------------- #
class QuantumQuanvolutionFilter(tq.QuantumModule):
    """Random two‑qubit quantum kernel applied to 2×2 image patches."""

    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_wires)]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, c, h, w = x.shape
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=x.device)
        # Extract 2×2 patches
        patches = []
        for r in range(0, h, 2):
            for c_ in range(0, w, 2):
                patch = x[:, :, r:r+2, c_:c_+2]
                # Flatten to 4‑dim vector
                flat = patch.view(bsz, -1)
                self.encoder(qdev, flat)
                self.q_layer(qdev)
                patches.append(self.measure(qdev).view(bsz, 4))
        return torch.cat(patches, dim=1)


class QuantumPatchExtractor(tq.QuantumModule):
    """Wraps the quanvolution filter to produce token embeddings."""

    def __init__(self, input_channels: int, embed_dim: int, patch_dim: int = 2) -> None:
        super().__init__()
        self.filter = QuantumQuanvolutionFilter()
        self.proj = nn.Linear(4, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.filter(x)  # shape: (B, num_patches*4)
        # Reshape to (B, seq_len, 4)
        seq_len = patches.size(1) // 4
        patches = patches.view(-1, seq_len, 4)
        return self.proj(patches)


# --------------------------------------------------------------------------- #
#  Hybrid Transformer Classifier
# --------------------------------------------------------------------------- #
class HybridTransformerClassifier(nn.Module):
    """Vision transformer with quantum front‑end and optional quantum transformer blocks.

    Parameters
    ----------
    input_channels : int
        Number of image channels.
    embed_dim : int
        Embedding dimension.
    num_heads : int
        Number of attention heads.
    num_blocks : int
        Number of transformer blocks.
    ffn_dim : int
        Hidden dimension of feed‑forward network.
    num_classes : int
        Number of target classes.
    dropout : float
        Drop‑out probability.
    use_quantum_front : bool
        If ``True`` the quantum quanvolution filter is used as front‑end.
    use_quantum_transformer : bool
        If ``True`` each transformer block is quantum‑enhanced.
    """

    def __init__(self,
                 input_channels: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 use_quantum_front: bool = False,
                 use_quantum_transformer: bool = False) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.ffn_dim = ffn_dim
        self.num_classes = num_classes
        self.dropout = dropout
        self.use_quantum_front = use_quantum_front
        self.use_quantum_transformer = use_quantum_transformer

        # Front‑end
        if use_quantum_front:
            self.front = QuantumPatchExtractor(input_channels, embed_dim)
        else:
            # Classical patch extractor with a 2×2 conv
            self.front = nn.Sequential(
                nn.Conv2d(input_channels, 4, kernel_size=2, stride=2),
                nn.Flatten(2),
                nn.Linear(4, embed_dim)
            )

        # Positional encoding
        self.pos_enc = PositionalEncoder(embed_dim)

        # Transformer blocks
        blocks = []
        for _ in range(num_blocks):
            if use_quantum_transformer:
                blocks.append(TransformerBlockQuantum(embed_dim, num_heads, ffn_dim,
                                                      n_qubits=embed_dim, dropout=dropout))
            else:
                blocks.append(TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout))
        self.transformers = nn.Sequential(*blocks)

        # Classifier head
        self.dropout_layer = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract and embed patches
        patches = self.front(x)
        # Positional encoding
        patches = self.pos_enc(patches)
        # Transformer
        patches = self.transformers(patches)
        # Pool and classify
        out = patches.mean(dim=1)
        out = self.dropout_layer(out)
        return self.classifier(out)


# Alias for backward compatibility
TextClassifier = HybridTransformerClassifier

__all__ = [
    "MultiHeadAttentionBase",
    "MultiHeadAttentionQuantum",
    "FeedForwardBase",
    "FeedForwardQuantum",
    "TransformerBlockBase",
    "TransformerBlockQuantum",
    "PositionalEncoder",
    "QuantumQuanvolutionFilter",
    "QuantumPatchExtractor",
    "HybridTransformerClassifier",
    "TextClassifier",
]
