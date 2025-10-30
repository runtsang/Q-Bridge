"""
Quantum‑enhanced transformer implementation using torchquantum.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq


class MultiHeadAttentionBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

    def attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        return torch.matmul(scores, value)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError


class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq, _ = x.shape
        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)
        k = k.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        q = q.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        attn = self.attention(q, k, v, mask)
        attn = attn.transpose(1, 2).contiguous().view(batch, seq, self.embed_dim)
        return self.out_linear(attn)


class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    """
    Quantum‑enhanced attention where each head projection is implemented
    by a parameterised quantum circuit. The circuit applies an RX gate
    per qubit followed by a fixed CNOT ladder and measurement of Pauli‑Z.
    """

    class _QuantumHead(tq.QuantumModule):
        def __init__(self, n_qubits: int):
            super().__init__()
            self.n_qubits = n_qubits
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
            )
            self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_qubits)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, dev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(dev, x)
            for i, gate in enumerate(self.params):
                gate(dev, wires=[i])
            return self.measure(dev)

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, n_qubits_per_head: Optional[int] = None) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.n_qubits_per_head = n_qubits_per_head or self.d_k
        self.q_heads = nn.ModuleList([self._QuantumHead(self.n_qubits_per_head) for _ in range(num_heads)])
        self.out_linear = nn.Linear(embed_dim, embed_dim)

    def _apply_quantum(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        heads = x.view(batch, seq, self.num_heads, self.d_k)
        outputs = []
        for h in range(self.num_heads):
            head_slice = heads[:, :, h, :]  # (batch, seq, d_k)
            flat = head_slice.reshape(-1, self.d_k)
            q_out = []
            for token in flat:
                dev = tq.QuantumDevice(n_wires=self.n_qubits_per_head, bsz=1, device=token.device)
                q_out.append(self.q_heads[h](token, dev))
            q_out = torch.stack(q_out).view(batch, seq, self.n_qubits_per_head)
            outputs.append(q_out)
        combined = torch.cat(outputs, dim=2).view(batch, seq, self.embed_dim)
        return combined

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        quantum_q = self._apply_quantum(x)
        quantum_k = self._apply_quantum(x)
        quantum_v = self._apply_quantum(x)
        batch, seq, _ = x.shape
        q = quantum_q.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        k = quantum_k.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        v = quantum_v.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        attn = self.attention(q, k, v, mask)
        attn = attn.transpose(1, 2).contiguous().view(batch, seq, self.embed_dim)
        return self.out_linear(attn)


class FeedForwardBase(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class FeedForwardQuantum(FeedForwardBase):
    """
    Quantum feed‑forward using a parameterised circuit per token.
    """

    class _QuantumFFN(tq.QuantumModule):
        def __init__(self, n_qubits: int):
            super().__init__()
            self.n_qubits = n_qubits
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
            )
            self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_qubits)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, dev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(dev, x)
            for i, gate in enumerate(self.params):
                gate(dev, wires=[i])
            return self.measure(dev)

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.q_ffn = self._QuantumFFN(embed_dim)
        self.q_device = tq.QuantumDevice(n_wires=embed_dim)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        q_outs = []
        for b in range(batch):
            for s in range(seq):
                token = x[b, s]
                dev = self.q_device.copy(bsz=1, device=token.device)
                q_outs.append(self.q_ffn(token, dev))
        q_out = torch.stack(q_outs).view(batch, seq, self.q_ffn.n_qubits)
        q_out = self.linear1(self.dropout(q_out))
        q_out = F.relu(q_out)
        q_out = self.linear2(q_out)
        return q_out


class TransformerBlockBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TransformerBlockQuantum(TransformerBlockBase):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        n_qubits_per_head: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout, n_qubits_per_head)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, dropout)

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


class QuantumTransformerEnhanced(nn.Module):
    """
    Quantum‑enhanced transformer.  The class name mirrors the classical
    implementation but the blocks contain quantum sub‑modules.
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
        n_qubits_per_head: int = 8,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlockQuantum(
                    embed_dim,
                    num_heads,
                    ffn_dim,
                    n_qubits_per_head,
                    dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        out_dim = num_classes if num_classes > 2 else 1
        self.classifier = nn.Linear(embed_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_encoder(tokens)
        x = self.transformer_blocks(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)


__all__ = [
    "MultiHeadAttentionBase",
    "MultiHeadAttentionClassical",
    "MultiHeadAttentionQuantum",
    "FeedForwardBase",
    "FeedForwardQuantum",
    "TransformerBlockBase",
    "TransformerBlockQuantum",
    "PositionalEncoder",
    "QuantumTransformerEnhanced",
]
