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


class MHA(nn.Module):
    """
    Efficient multi‑head attention (classical fallback).
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2), float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, T, self.embed_dim)
        return self.out_proj(out)


class FeedForward(nn.Module):
    """
    Two‑layer perceptron feed‑forward network (classical fallback).
    """

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QuantumAttention(nn.Module):
    """
    Multi‑head attention that maps projections through quantum modules.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.q_enc = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(self.head_dim)]
        )
        self.k_enc = self.q_enc
        self.v_enc = self.q_enc
        self.q_gate = nn.ModuleList(
            [tq.RX(has_params=True, trainable=True) for _ in range(self.head_dim)]
        )
        self.k_gate = self.q_gate
        self.v_gate = self.q_gate
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.size()
        x = x.view(B, T, self.num_heads, self.head_dim)
        out_q, out_k, out_v = [], [], []
        for i in range(self.num_heads):
            qdev = tq.QuantumDevice(
                n_wires=self.head_dim, bsz=B * T, device=x.device
            )
            self.q_enc(qdev, x[:, :, i, :].reshape(-1, self.head_dim))
            for gate in self.q_gate:
                gate(qdev, wires=range(self.head_dim))
            out_q.append(self.measure(qdev).reshape(B, T, self.head_dim))
            self.k_enc(qdev, x[:, :, i, :].reshape(-1, self.head_dim))
            for gate in self.k_gate:
                gate(qdev, wires=range(self.head_dim))
            out_k.append(self.measure(qdev).reshape(B, T, self.head_dim))
            self.v_enc(qdev, x[:, :, i, :].reshape(-1, self.head_dim))
            for gate in self.v_gate:
                gate(qdev, wires=range(self.head_dim))
            out_v.append(self.measure(qdev).reshape(B, T, self.head_dim))
        q = torch.stack(out_q, dim=2)
        k = torch.stack(out_k, dim=2)
        v = torch.stack(out_v, dim=2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2), float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.view(B, T, self.embed_dim)
        return self.proj(out)


class QuantumFeedForward(nn.Module):
    """
    Feed‑forward network realized by a quantum module.
    """

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(embed_dim)]
        )
        self.params = nn.ModuleList(
            [tq.RY(has_params=True, trainable=True) for _ in range(embed_dim)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.lin1 = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.lin2 = nn.Linear(ffn_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        out = []
        for i in range(B * T):
            qdev = tq.QuantumDevice(n_wires=C, bsz=1, device=x.device)
            self.encoder(qdev, x[i].unsqueeze(0))
            for gate in self.params:
                gate(qdev, wires=range(C))
            out.append(self.measure(qdev).squeeze(0))
        out = torch.stack(out, dim=0).view(B, T, C)
        out = self.lin1(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.lin2(out)
        return out


class ClassicalTransformerBlock(nn.Module):
    """
    Classical transformer block used as a fallback for hybrid mode.
    """

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MHA(embed_dim, num_heads, dropout)
        self.ffn = FeedForward(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class QuantumTransformerBlock(nn.Module):
    """
    Transformer block that uses quantum attention and feed‑forward layers.
    """

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = QuantumAttention(embed_dim, num_heads, dropout)
        self.ffn = QuantumFeedForward(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.
    """

    def __init__(self, embed_dim: int, max_len: int = 5_000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float) * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class QuantumTransformerDual(nn.Module):
    """
    Hybrid transformer that can operate in classical or quantum mode.
    The mode is chosen at construction time; the quantum implementation uses
    parameter‑shared circuits for attention and feed‑forward layers.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        mode: str = "classical",
    ) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_enc = PositionalEncoding(embed_dim)
        if mode == "quantum":
            self.blocks = nn.ModuleList(
                [
                    QuantumTransformerBlock(embed_dim, num_heads, ffn_dim, dropout)
                    for _ in range(num_layers)
                ]
            )
        else:
            self.blocks = nn.ModuleList(
                [
                    ClassicalTransformerBlock(embed_dim, num_heads, ffn_dim, dropout)
                    for _ in range(num_layers)
                ]
            )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(
            embed_dim, num_classes if num_classes > 2 else 1
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.token_emb(x)
        x = self.pos_enc(x)
        for blk in self.blocks:
            x = blk(x, mask)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)


__all__ = [
    "QuantumAttention",
    "QuantumFeedForward",
    "QuantumTransformerBlock",
    "PositionalEncoding",
    "QuantumTransformerDual",
]
