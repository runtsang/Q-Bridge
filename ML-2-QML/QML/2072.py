"""
Quantum‑enhanced transformer implementation based on torchquantum.
Class name is HybridTransformer to match the classical counterpart.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class MultiHeadAttentionBase(nn.Module):
    """Shared logic for attention layers."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
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
        return self.dropout(scores), torch.matmul(scores, value)

class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    """Quantum‑parameterized attention."""
    def __init__(self, embed_dim: int, num_heads: int, n_wires: int,
                 dropout: float = 0.1, q_device: Optional[tq.QuantumDevice] = None):
        super().__init__(embed_dim, num_heads, dropout)
        self.n_wires = n_wires
        self.q_proj = nn.Linear(embed_dim, n_wires, bias=False)
        self.out_proj = nn.Linear(n_wires, embed_dim, bias=False)
        self.q_params = nn.Parameter(torch.randn(n_wires))
        self.cnot_chain = [(i, i + 1) for i in range(n_wires - 1)] + [(n_wires - 1, 0)]
        self.q_device = q_device or tq.QuantumDevice(n_wires=n_wires)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = self.q_proj(x)  # (B, T, n_wires)
        outputs = []
        for token in q.unbind(dim=1):  # iterate over sequence dimension
            qdev = self.q_device.copy(bsz=token.size(0), device=token.device)
            # Encode token values into rotations
            for idx in range(self.n_wires):
                tqf.rx(qdev, token[:, idx], wires=[idx])
            # Apply trainable rotations
            for idx, param in enumerate(self.q_params):
                tqf.rx(qdev, param, wires=[idx])
            # Entangle
            for src, tgt in self.cnot_chain:
                tqf.cnot(qdev, wires=[src, tgt])
            # Measure
            outputs.append(tqf.measure_all(qdev, wires=list(range(self.n_wires))))
        q_out = torch.stack(outputs, dim=1)  # (B, T, n_wires)
        return self.out_proj(q_out)

class FeedForwardBase(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

class FeedForwardQuantum(FeedForwardBase):
    """Feed‑forward layer realized by a variational quantum circuit."""
    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int,
                 dropout: float = 0.1):
        super().__init__(embed_dim, ffn_dim, dropout)
        self.n_qubits = n_qubits
        self.q_params = nn.Parameter(torch.randn(n_qubits))
        self.cnot_chain = [(i, i + 1) for i in range(n_qubits - 1)] + [(n_qubits - 1, 0)]
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim, bias=False)
        self.linear2 = nn.Linear(ffn_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for token in x.unbind(dim=1):
            qdev = self.q_device.copy(bsz=token.size(0), device=token.device)
            # Encode token values
            for idx in range(self.n_qubits):
                tqf.rx(qdev, token[:, idx], wires=[idx])
            # Apply trainable rotations
            for idx, param in enumerate(self.q_params):
                tqf.rx(qdev, param, wires=[idx])
            # Entangle
            for src, tgt in self.cnot_chain:
                tqf.cnot(qdev, wires=[src, tgt])
            # Measure
            outputs.append(tqf.measure_all(qdev, wires=list(range(self.n_qubits))))
        q_out = torch.stack(outputs, dim=1)  # (B, T, n_qubits)
        out = self.linear1(q_out)
        out = self.linear2(F.relu(self.dropout(out)))
        return out

class TransformerBlockBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

class TransformerBlockClassical(TransformerBlockBase):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
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
                 n_wires_attn: int, n_wires_ffn: int,
                 n_qlayers: int, dropout: float = 0.1):
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, n_wires_attn, dropout)
        if n_wires_ffn > 0:
            self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_wires_ffn, dropout)
        else:
            self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class HybridTransformer(nn.Module):
    """
    Quantum‑enhanced transformer that mirrors the classical API.
    If n_qubits_transformer or n_qubits_ffn are zero the block defaults to classical.
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
        n_qubits_transformer: int = 0,
        n_qubits_ffn: int = 0,
        n_qlayers: int = 1,
        q_device: Optional[tq.QuantumDevice] = None,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        if n_qubits_transformer > 0:
            self.transformer_layers = nn.ModuleList(
                [
                    TransformerBlockQuantum(
                        embed_dim,
                        num_heads,
                        ffn_dim,
                        n_qubits_transformer,
                        n_qubits_ffn,
                        n_qlayers,
                        dropout=dropout,
                    )
                    for _ in range(num_blocks)
                ]
            )
        else:
            self.transformer_layers = nn.ModuleList(
                [
                    TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
                    for _ in range(num_blocks)
                ]
            )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        x = self.pos_encoder(x)
        for layer in self.transformer_layers:
            x = layer(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)

__all__ = ["HybridTransformer"]
