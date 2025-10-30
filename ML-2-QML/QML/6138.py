"""Hybrid transformer with quantum‑enhanced modules and regression head."""
from __future__ import annotations

import math
from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class MultiHeadAttention(nn.Module):
    """Classical multi‑head attention."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq_len, _ = x.size()
        qkv = self.qkv_proj(x).reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        scores = torch.einsum('bnhd,bmhd->bhnm', q, k) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.einsum('bhnm,bmhd->bnhd', attn, v)
        out = out.reshape(batch, seq_len, self.embed_dim)
        return self.out_proj(out)


class FeedForwardQuantum(nn.Module):
    """Quantum feed‑forward block inspired by EstimatorQNN."""
    class QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 1
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [0], "func": "ry", "wires": [0]}]
            )
            self.rx_gate = tq.RX(has_params=True, trainable=True)
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            self.rx_gate(q_device, wires=0)
            return self.measure(q_device)

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.q_layer = self.QLayer()
        self.q_device = tq.QuantumDevice(n_wires=1)
        self.linear1 = nn.Linear(embed_dim, 1)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for token in x.unbind(dim=1):
            scalar = self.linear1(token).unsqueeze(-1)  # (batch,1)
            qdev = self.q_device.copy(bsz=scalar.size(0), device=scalar.device)
            qout = self.q_layer(scalar, qdev)
            outputs.append(qout.unsqueeze(1))
        out = torch.cat(outputs, dim=1)  # (batch, seq_len, 1)
        out = self.linear2(self.dropout(out))
        return out


class FeedForward(nn.Module):
    """Fallback classical feed‑forward."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Transformer block that can use quantum feed‑forward."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 use_quantum_ffn: bool = False, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, dropout) if use_quantum_ffn else FeedForward(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(self.norm1(x))
        x = x + self.dropout(attn_out)
        ffn_out = self.ffn(self.norm2(x))
        return x + self.dropout(ffn_out)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float32) *
                             (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class HybridTransformer(nn.Module):
    """Quantum‑enhanced transformer with optional regression head."""
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        regression: bool = False,
        quantum_ffn: bool = False,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim)
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, ffn_dim,
                              use_quantum_ffn=quantum_ffn, dropout=dropout)
             for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)
        self.regression_head = None
        if regression:
            class QuantumRegressor(tq.QuantumModule):
                def __init__(self):
                    super().__init__()
                    self.n_wires = 1
                    self.encoder = tq.GeneralEncoder(
                        [{"input_idx": [0], "func": "ry", "wires": [0]}]
                    )
                    self.rx_gate = tq.RX(has_params=True, trainable=True)
                    self.measure = tq.MeasureAll(tq.PauliZ)

                def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
                    self.encoder(q_device, x)
                    self.rx_gate(q_device, wires=0)
                    return self.measure(q_device)

            self.q_regressor = QuantumRegressor()
            self.q_device = tq.QuantumDevice(n_wires=1)
            self.linear_reg = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor, head: Literal['classify','regress'] = 'classify') -> torch.Tensor:
        x = self.token_embedding(x)
        x = self.pos_encoding(x)
        for block in self.blocks:
            x = block(x)
        x = self.dropout(x.mean(dim=1))
        if head == 'classify':
            return self.classifier(x)
        elif head =='regress':
            if self.regression_head is None:
                raise ValueError("Regression head not initialized")
            qdev = self.q_device.copy(bsz=x.size(0), device=x.device)
            qout = self.q_regressor(x.mean(dim=1).unsqueeze(-1), qdev)
            return self.linear_reg(qout)
        else:
            raise ValueError(f"Unknown head {head}")

__all__ = ['HybridTransformer']
