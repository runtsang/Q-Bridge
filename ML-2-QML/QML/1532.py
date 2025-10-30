"""Quantum‑enhanced Transformer implementation for hybrid experiments.

Classes
-------
QLayer : variational quantum circuit used in attention and feed‑forward.
MultiHeadAttentionHybridBase : shared base for attention layers.
MultiHeadAttentionHybrid : classical attention (API compatibility).
MultiHeadAttentionQuantumHybrid : attention with quantum‑augmented projections.
FeedForwardHybridBase : base feed‑forward layer.
FeedForwardHybrid : classical feed‑forward.
FeedForwardQuantumHybrid : feed‑forward with quantum post‑processing.
TransformerBlockHybridBase : base transformer block.
TransformerBlockHybrid : classical block.
TransformerBlockQuantumHybrid : block that replaces attention and feed‑forward with quantum variants.
PositionalEncodingLearned : learned positional encoder.
TextClassifierHybrid : main classifier that can toggle quantum sub‑modules.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


def _check_embed_dim(embed_dim: int, num_heads: int) -> None:
    if embed_dim % num_heads!= 0:
        raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")


# --------------------------------------------------------------------------- #
# Variational quantum layer
# --------------------------------------------------------------------------- #
class QLayer(tq.QuantumModule):
    """Small variational circuit that maps an n_qubit input to an n_qubit output."""

    def __init__(self, n_qubits: int, n_layers: int = 2) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
        )
        self.parameters = nn.ModuleList(
            [tq.RY(has_params=True, trainable=True) for _ in range(n_qubits * n_layers)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        # x shape: (batch, n_qubits)
        self.encoder(q_device, x)
        idx = 0
        for _ in range(self.n_layers):
            for wire in range(self.n_qubits):
                self.parameters[idx](q_device, wires=wire)
                idx += 1
        for i in range(self.n_qubits - 1):
            tqf.cnot(q_device, wires=[i, i + 1])
        return self.measure(q_device)


# --------------------------------------------------------------------------- #
# Multi‑head attention
# --------------------------------------------------------------------------- #
class MultiHeadAttentionHybridBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        _check_embed_dim(embed_dim, num_heads)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError


class MultiHeadAttentionHybrid(MultiHeadAttentionHybridBase):
    """Classical multi‑head attention (API compatibility)."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq_len, _ = x.size()
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
        return self.out_proj(out)


class MultiHeadAttentionQuantumHybrid(MultiHeadAttentionHybridBase):
    """Attention layer that augments each linear projection with a variational quantum circuit."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        n_qubits: int = 8,
        n_layers: int = 2,
    ) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Linear maps from d_k to n_qubits and back
        self.project_in = nn.ModuleList([nn.Linear(self.d_k, n_qubits, bias=False) for _ in range(3)])  # q, k, v
        self.project_out = nn.ModuleList([nn.Linear(n_qubits, self.d_k, bias=False) for _ in range(3)])

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.q_layer = QLayer(n_qubits, n_layers)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq_len, _ = x.size()
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        q_trans, k_trans, v_trans = [], [], []
        for i in range(self.num_heads):
            q_head = q[:, i].reshape(batch * seq_len, self.d_k)
            k_head = k[:, i].reshape(batch * seq_len, self.d_k)
            v_head = v[:, i].reshape(batch * seq_len, self.d_k)

            q_proj = self.project_in[0](q_head)
            k_proj = self.project_in[1](k_head)
            v_proj = self.project_in[2](v_head)

            q_device = tq.QuantumDevice(n_wires=self.n_qubits, bsz=q_proj.size(0), device=x.device)
            k_device = tq.QuantumDevice(n_wires=self.n_qubits, bsz=k_proj.size(0), device=x.device)
            v_device = tq.QuantumDevice(n_wires=self.n_qubits, bsz=v_proj.size(0), device=x.device)

            q_q = self.q_layer(q_proj, q_device)
            k_q = self.q_layer(k_proj, k_device)
            v_q = self.q_layer(v_proj, v_device)

            q_q = q_q.view(batch, seq_len, self.n_qubits)
            k_q = k_q.view(batch, seq_len, self.n_qubits)
            v_q = v_q.view(batch, seq_len, self.n_qubits)

            q_q = self.project_out[0](q_q)
            k_q = self.project_out[1](k_q)
            v_q = self.project_out[2](v_q)

            q_trans.append(q_q)
            k_trans.append(k_q)
            v_trans.append(v_q)

        q = torch.stack(q_trans, dim=1)
        k = torch.stack(k_trans, dim=1)
        v = torch.stack(v_trans, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
        return self.out_proj(out)


# --------------------------------------------------------------------------- #
# Feed‑forward
# --------------------------------------------------------------------------- #
class FeedForwardHybridBase(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class FeedForwardHybrid(FeedForwardHybridBase):
    """Classical feed‑forward."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.fc1 = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.fc2 = nn.Linear(ffn_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class FeedForwardQuantumHybrid(FeedForwardHybridBase):
    """Feed‑forward that adds a quantum post‑processing step."""
    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        n_qubits: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.project_in = nn.Linear(embed_dim, n_qubits, bias=False)
        self.project_out = nn.Linear(n_qubits, embed_dim, bias=False)
        self.q_layer = QLayer(n_qubits, n_layers)
        self.fc1 = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.fc2 = nn.Linear(ffn_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.size()
        q_in = self.project_in(x.reshape(-1, x.size(-1)))
        q_device = tq.QuantumDevice(n_wires=self.q_layer.n_qubits, bsz=q_in.size(0), device=x.device)
        q_out = self.q_layer(q_in, q_device).reshape(batch, seq_len, -1)
        x_q = self.project_out(q_out)

        return self.fc2(self.dropout(F.relu(self.fc1(x_q))))


# --------------------------------------------------------------------------- #
# Transformer block
# --------------------------------------------------------------------------- #
class TransformerBlockHybridBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TransformerBlockHybrid(TransformerBlockHybridBase):
    """Classical transformer block."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionHybrid(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardHybrid(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class TransformerBlockQuantumHybrid(TransformerBlockHybridBase):
    """Transformer block that replaces attention and feed‑forward with quantum variants."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        n_qubits_attn: int = 8,
        n_layers_attn: int = 2,
        n_qubits_ffn: int = 8,
        n_layers_ffn: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionQuantumHybrid(
            embed_dim, num_heads, dropout, n_qubits_attn, n_layers_attn
        )
        self.ffn = FeedForwardQuantumHybrid(
            embed_dim, ffn_dim, n_qubits_ffn, n_layers_ffn, dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


# --------------------------------------------------------------------------- #
# Positional encoding
# --------------------------------------------------------------------------- #
class PositionalEncodingLearned(nn.Module):
    """Learned positional encoding implemented as an MLP."""
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        self.position_embed = nn.Embedding(max_len, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(x.size(0), seq_len)
        pos_emb = self.position_embed(pos)
        return x + self.mlp(pos_emb)


# --------------------------------------------------------------------------- #
# Text classifier
# --------------------------------------------------------------------------- #
class TextClassifierHybrid(nn.Module):
    """Transformer‑based classifier that can toggle quantum sub‑modules."""
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
        n_qubits_attn: int = 8,
        n_layers_attn: int = 2,
        n_qubits_ffn: int = 8,
        n_layers_ffn: int = 2,
        **quantum_kwargs,
    ) -> None:
        super().__init__()
        self.use_quantum = use_quantum
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncodingLearned(embed_dim)

        blocks = []
        for _ in range(num_blocks):
            if use_quantum:
                blocks.append(
                    TransformerBlockQuantumHybrid(
                        embed_dim,
                        num_heads,
                        ffn_dim,
                        n_qubits_attn,
                        n_layers_attn,
                        n_qubits_ffn,
                        n_layers_ffn,
                        dropout,
                    )
                )
            else:
                blocks.append(
                    TransformerBlockHybrid(
                        embed_dim,
                        num_heads,
                        ffn_dim,
                        dropout,
                    )
                )
        self.transformer = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        x = self.pos_embedding(x)
        x = self.transformer(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)


__all__ = [
    "QLayer",
    "MultiHeadAttentionHybridBase",
    "MultiHeadAttentionHybrid",
    "MultiHeadAttentionQuantumHybrid",
    "FeedForwardHybridBase",
    "FeedForwardHybrid",
    "FeedForwardQuantumHybrid",
    "TransformerBlockHybridBase",
    "TransformerBlockHybrid",
    "TransformerBlockQuantumHybrid",
    "PositionalEncodingLearned",
    "TextClassifierHybrid",
]
