"""Quantum‑enhanced transformer implementation using TorchQuantum."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding (identical to classical)."""
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


class QModule(tq.QuantumModule):
    """Simple variational circuit that encodes a classical vector into qubit amplitudes."""
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
        )
        self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
        self.measure = tq.MeasureAll(tq.PauliZ)
    def forward(self, input_vec: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
        self.encoder(qdev, input_vec)
        for w, gate in enumerate(self.params):
            gate(qdev, wires=w)
        return self.measure(qdev)


class MultiHeadAttention(nn.Module):
    """Attention that can be classical, quantum, or hybrid."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, mode: str = "quantum"):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.mode = mode
        if mode == "classical":
            self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
            self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
            self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)
            self.out_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        elif mode in ("quantum", "hybrid"):
            self.n_wires = self.d_k
            self.q_module = QModule(self.n_wires)
        else:
            raise ValueError("mode must be 'classical', 'quantum', or 'hybrid'")
    def _reshape(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, dim = x.size()
        return x.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq, dim = x.size()
        if self.mode == "classical":
            q = self._reshape(self.q_linear(x))
            k = self._reshape(self.k_linear(x))
            v = self._reshape(self.v_linear(x))
        else:
            q_list, k_list, v_list = [], [], []
            qdev = self._get_qdevice(batch, x.device)
            for token in x.unbind(dim=1):
                token_slice = token[:, :self.n_wires]
                q = self.q_module(token_slice, qdev)
                k = self.q_module(token_slice, qdev)
                v = self.q_module(token_slice, qdev)
                q_list.append(q)
                k_list.append(k)
                v_list.append(v)
            q = torch.stack(q_list, dim=1)
            k = torch.stack(k_list, dim=1)
            v = torch.stack(v_list, dim=1)
            q = self._reshape(q)
            k = self._reshape(k)
            v = self._reshape(v)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq, self.embed_dim)
        if self.mode == "classical":
            return self.out_linear(out)
        else:
            return out
    def _get_qdevice(self, batch: int, device: torch.device) -> tq.QuantumDevice:
        return tq.QuantumDevice(n_wires=self.n_wires, bsz=batch, device=device)


class FeedForward(nn.Module):
    """Feed‑forward that can be classical or quantum."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1, mode: str = "quantum"):
        super().__init__()
        self.mode = mode
        if mode == "classical":
            self.linear1 = nn.Linear(embed_dim, ffn_dim)
            self.linear2 = nn.Linear(ffn_dim, embed_dim)
        elif mode in ("quantum", "hybrid"):
            self.n_wires = ffn_dim
            self.q_module = QModule(self.n_wires)
            self.linear1 = nn.Linear(ffn_dim, embed_dim)
            self.linear2 = nn.Linear(embed_dim, embed_dim)
        else:
            raise ValueError("mode must be 'classical', 'quantum', or 'hybrid'")
        self.dropout = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, dim = x.size()
        if self.mode == "classical":
            return self.linear2(self.dropout(F.relu(self.linear1(x))))
        else:
            out_list = []
            qdev = self._get_qdevice(batch, x.device)
            for token in x.unbind(dim=1):
                token_slice = token[:, :self.n_wires]
                q = self.q_module(token_slice, qdev)
                out_list.append(q)
            out = torch.stack(out_list, dim=1)
            out = self.linear1(self.dropout(out))
            return self.linear2(F.relu(out))
    def _get_qdevice(self, batch: int, device: torch.device) -> tq.QuantumDevice:
        return tq.QuantumDevice(n_wires=self.n_wires, bsz=batch, device=device)


class TransformerBlock(nn.Module):
    """Transformer block that can mix classical and quantum sub‑modules."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1, mode: str = "quantum"):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout, mode)
        self.ffn = FeedForward(embed_dim, ffn_dim, dropout, mode)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.attn(x)))
        return self.norm2(x + self.dropout(self.ffn(x)))


class QuantumTransformerHybrid(nn.Module):
    """Transformer that supports classical, quantum, or hybrid modes."""
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        mode: str = "quantum",
    ) -> None:
        super().__init__()
        if mode not in ("classical", "quantum", "hybrid"):
            raise ValueError("mode must be 'classical', 'quantum', or 'hybrid'")
        self.mode = mode
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, ffn_dim, dropout, mode) for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        x = self.pos_encoder(x)
        for block in self.blocks:
            x = block(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)


__all__ = [
    "PositionalEncoder",
    "QModule",
    "MultiHeadAttention",
    "FeedForward",
    "TransformerBlock",
    "QuantumTransformerHybrid",
]
