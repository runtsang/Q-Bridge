"""Quantum‑enhanced transformer with variational circuits for attention and feed‑forward."""

from __future__ import annotations

import math
from typing import Optional, Sequence, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class MultiHeadAttentionClassical(nn.Module):
    """Standard multi‑head attention implemented with linear projections."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        return x.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                  mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        return self.dropout(scores) @ v, scores

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        proj = self.proj(x)
        q, k, v = proj, proj, proj
        q = self.separate_heads(q)
        k = self.separate_heads(k)
        v = self.separate_heads(v)
        attn, _ = self.attention(q, k, v, mask)
        return self.out_proj(attn.transpose(1, 2).contiguous().view(x.shape))


class MultiHeadAttentionQuantum(nn.Module):
    """Quantum‑augmented multi‑head attention using a variational ansatz."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, n_wires: int = 8):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.n_wires = n_wires
        self.dropout = nn.Dropout(dropout)

        # Classical linear projections
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.qc = self._build_circuit()

    def _build_circuit(self):
        def circuit(qdev: tq.QuantumDevice, x: torch.Tensor) -> torch.Tensor:
            for i in range(self.n_wires):
                tqf.rx(qdev, wires=[i], params=x[:, i])
            for i in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[i, i + 1])
            return tqf.measure_all(qdev)
        return circuit

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq, _ = x.shape
        k = self.k_proj(x)
        q = self.q_proj(x)
        v = self.v_proj(x)

        k = k.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        q = q.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)

        out_heads = []
        for h in range(self.num_heads):
            head_q = q[:, h]  # (batch, seq, d_k)
            flat = head_q.reshape(batch * seq, self.d_k)
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=batch * seq, device=head_q.device)
            out_q = self.qc(qdev, flat)
            out_q = out_q.reshape(batch, seq, self.d_k)
            out_heads.append(out_q)
        out_heads = torch.stack(out_heads, dim=1)
        out = out_heads.transpose(1, 2).contiguous().view(batch, seq, self.embed_dim)
        return self.out_proj(out)


class FeedForwardQuantum(nn.Module):
    """Feed‑forward network realised by a variational quantum circuit."""
    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.n_qubits = n_qubits
        self.dropout = nn.Dropout(dropout)

        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

        self.qc = self._build_circuit()

    def _build_circuit(self):
        def circuit(qdev: tq.QuantumDevice, x: torch.Tensor) -> torch.Tensor:
            for i in range(self.n_qubits):
                tqf.rx(qdev, wires=[i], params=x[:, i])
            for i in range(self.n_qubits - 1):
                tqf.cnot(qdev, wires=[i, i + 1])
            return tqf.measure_all(qdev)
        return circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        flat = x.reshape(batch * seq, self.n_qubits)
        qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=batch * seq, device=x.device)
        out_q = self.qc(qdev, flat)
        out_q = out_q.reshape(batch, seq, self.n_qubits)
        out_q = self.linear1(self.dropout(out_q))
        return self.linear2(F.relu(out_q))


class TransformerBlockBase(nn.Module):
    """Base transformer block with residual connections."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TransformerBlockQuantum(TransformerBlockBase):
    """Transformer block that uses quantum attention and feed‑forward."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 n_qubits: int = 8, dropout: float = 0.1):
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout, n_wires=n_qubits)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class TransformerBlockClassical(TransformerBlockBase):
    """Classical transformer block for fallback."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, dropout=dropout)  # reuse quantum ffn for consistency

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class HybridTransformer(nn.Module):
    """Hybrid transformer that can interleave classical and quantum blocks."""
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_blocks: int,
        num_classes: int,
        dropout: float = 0.1,
        block_types: Sequence[str] = ("classical",) * 6,
        n_qubits: int = 8,
    ):
        super().__init__()
        if len(block_types)!= num_blocks:
            raise ValueError("block_types length must match num_blocks")
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)

        blocks: List[nn.Module] = []
        for btype in block_types:
            if btype == "classical":
                blocks.append(TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout))
            elif btype == "quantum":
                blocks.append(TransformerBlockQuantum(embed_dim, num_heads, ffn_dim, n_qubits, dropout))
            else:
                raise ValueError(f"Unknown block type: {btype}")

        self.transformer = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)
