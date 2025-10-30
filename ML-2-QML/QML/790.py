"""
Quantum‑enhanced transformer layers implemented with TorchQuantum.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


# ----------------------------------------------------------------------
# Base attention and feed‑forward layers
# ----------------------------------------------------------------------
class MultiHeadAttentionBase(nn.Module):
    """
    Base class for all attention variants.
    Keeps the same API as the seed, but adds a *quantum‑aware* dropout
    that can be tuned during training.
    """
    def __init__(self, embed_dim: int, num_heads: int,
                 dropout: float = 0.1, use_bias: bool = False) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.qdrop_scale = nn.Parameter(torch.tensor(dropout, dtype=torch.float32))
        self.use_bias = use_bias

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape into (batch, heads, seq, d_k)."""
        batch_size = x.size(0)
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

    def attention(self, query: torch.Tensor, key: torch.Tensor,
                  value: torch.Tensor, mask: Optional[torch.Tensor] = None
                 ) -> tuple[torch.Tensor, torch.Tensor]:
        """Scaled dot‑product attention with optional mask."""
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = mask.unsqueeze(1).masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        return torch.matmul(scores, value), scores


# ----------------------------------------------------------------------
# Quantum modules
# ----------------------------------------------------------------------
class QuantumAttention(tq.QuantumModule):
    """
    Simple variational quantum circuit that acts on each attention head.
    """
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

    def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        self.encoder(q_device, x)
        for wire, gate in enumerate(self.parameters):
            gate(q_device, wires=wire)
        return self.measure(q_device)


class QuantumFeedForward(tq.QuantumModule):
    """
    Variational circuit that replaces the first linear layer of the feed‑forward block.
    """
    def __init__(self, n_qubits: int) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
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


# ----------------------------------------------------------------------
# Classical and hybrid attention
# ----------------------------------------------------------------------
class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """
    Standard multi‑head attention implemented classically.
    """
    def __init__(self, embed_dim: int, num_heads: int,
                 dropout: float = 0.1, use_bias: bool = False) -> None:
        super().__init__(embed_dim, num_heads, dropout, use_bias)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=use_bias)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq, _ = x.size()
        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)
        q = self.separate_heads(q)
        k = self.separate_heads(k)
        v = self.separate_heads(v)
        attn_output, _ = self.attention(q, k, v, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq, self.embed_dim)
        return self.out_proj(attn_output)


class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    """
    Multi‑head attention that maps projections through a quantum circuit.
    """
    def __init__(self, embed_dim: int, num_heads: int,
                 dropout: float = 0.1, use_bias: bool = False,
                 q_device: Optional[tq.QuantumDevice] = None) -> None:
        super().__init__(embed_dim, num_heads, dropout, use_bias)
        self.q_layer = QuantumAttention()
        self.q_device = q_device or tq.QuantumDevice(n_wires=self.q_layer.n_wires)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=use_bias)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq, _ = x.size()
        # Classical projections
        k = self.q_layer.encoder(self.q_device, x).reshape(batch, seq, self.num_heads, self.d_k)
        q = self.q_layer.encoder(self.q_device, x).reshape(batch, seq, self.num_heads, self.d_k)
        v = self.q_layer.encoder(self.q_device, x).reshape(batch, seq, self.num_heads, self.d_k)
        # Apply quantum circuit per head
        outputs = []
        for h in range(self.num_heads):
            head = torch.stack([self.q_layer(x[:, i, h, :], self.q_device)
                                for i in range(seq)], dim=1)
            outputs.append(head)
        attn_output = torch.stack(outputs, dim=1).contiguous()
        attn_output = attn_output.view(batch, seq, self.embed_dim)
        return self.out_proj(attn_output)


# ----------------------------------------------------------------------
# Classical and hybrid feed‑forward
# ----------------------------------------------------------------------
class FeedForwardClassical(FeedForwardBase):
    """
    Two‑layer perceptron feed‑forward network.
    """
    def __init__(self, embed_dim: int, ffn_dim: int,
                 dropout: float = 0.1, use_bias: bool = False) -> None:
        super().__init__(embed_dim, ffn_dim, dropout, use_bias)
        self.linear1 = nn.Linear(embed_dim, ffn_dim, bias=use_bias)
        self.linear2 = nn.Linear(ffn_dim, embed_dim, bias=use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class FeedForwardQuantum(FeedForwardBase):
    """
    Feed‑forward network realised by a quantum module.
    """
    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int,
                 dropout: float = 0.1, use_bias: bool = False) -> None:
        super().__init__(embed_dim, ffn_dim, dropout, use_bias)
        self.q_layer = QuantumFeedForward(n_qubits)
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim, bias=use_bias)
        self.linear2 = nn.Linear(ffn_dim, embed_dim, bias=use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for token in x.unbind(dim=1):
            qdev = self.q_device.copy(bsz=token.size(0), device=token.device)
            outputs.append(self.q_layer(token, qdev))
        out = torch.stack(outputs, dim=1)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))


# ----------------------------------------------------------------------
# Transformer blocks
# ----------------------------------------------------------------------
class TransformerBlockClassical(TransformerBlockBase):
    """
    Classical transformer block.
    """
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 dropout: float = 0.1, use_bias: bool = False) -> None:
        super().__init__(embed_dim, num_heads, dropout, use_bias)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads,
                                                dropout, use_bias)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim,
                                        dropout, use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class TransformerBlockQuantum(TransformerBlockBase):
    """
    Transformer block that uses quantum attention and feed‑forward.
    """
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 n_qubits_transformer: int, n_qubits_ffn: int,
                 q_device: Optional[tq.QuantumDevice] = None,
                 dropout: float = 0.1, use_bias: bool = False) -> None:
        super().__init__(embed_dim, num_heads, dropout, use_bias)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads,
                                              dropout, use_bias, q_device)
        if n_qubits_ffn > 0:
            self.ffn = FeedForwardQuantum(embed_dim, ffn_dim,
                                          n_qubits_ffn, dropout, use_bias)
        else:
            self.ffn = FeedForwardClassical(embed_dim, ffn_dim,
                                            dropout, use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


# ----------------------------------------------------------------------
# Positional encoding
# ----------------------------------------------------------------------
class PositionalEncoder(nn.Module):
    """
    Sinusoidal positional encoding.
    """
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


# ----------------------------------------------------------------------
# Text classifier
# ----------------------------------------------------------------------
class TextClassifier(nn.Module):
    """
    Transformer‑based text classifier supporting quantum sub‑modules.
    """
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 use_bias: bool = False,
                 n_qubits_transformer: int = 0,
                 n_qubits_ffn: int = 0,
                 q_device: Optional[tq.QuantumDevice] = None) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        if n_qubits_transformer > 0:
            q_device = q_device or tq.QuantumDevice(
                n_wires=max(n_qubits_transformer, n_qubits_ffn)
            )
            blocks = [
                TransformerBlockQuantum(
                    embed_dim, num_heads, ffn_dim,
                    n_qubits_transformer, n_qubits_ffn,
                    q_device=q_device, dropout=dropout, use_bias=use_bias
                )
                for _ in range(num_blocks)
            ]
        else:
            blocks = [
                TransformerBlockClassical(embed_dim, num_heads, ffn_dim,
                                         dropout, use_bias)
                for _ in range(num_blocks)
            ]
        self.transformers = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout)
        if num_classes > 2:
            self.classifier = nn.Linear(embed_dim, num_classes)
        else:
            self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_encoder(tokens)
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
