"""Quantum‑enhanced transformer using TorchQuantum."""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class PositionalEncoding(nn.Module):
    """Learnable sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000, learnable: bool = False):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
        if learnable:
            self.pe = nn.Parameter(self.pe.clone().detach())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class MultiHeadAttentionBase(nn.Module):
    """Base class for multi‑head attention variants."""
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

    def attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                  mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
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
    """Multi‑head attention with quantum projections."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int):
            super().__init__()
            self.n_qubits = n_qubits
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
            )
            self.parameters = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_qubits)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for i, gate in enumerate(self.parameters):
                gate(q_device, wires=i)
            return self.measure(q_device)

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__(embed_dim, num_heads, dropout)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.combine_heads = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_layer = self.QLayer(self.d_k)
        # quantum device will be created on‑the‑fly

    def _quantum_proj(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_k)
        batch, seq_len, d_k = x.shape
        flat = x.reshape(-1, d_k)
        qdev = tq.QuantumDevice(n_wires=self.d_k, bsz=flat.shape[0], device=flat.device)
        out = self.q_layer(flat, qdev)
        return out.reshape(batch, seq_len, d_k)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)

        # quantum projections
        k_q = self._quantum_proj(k)
        q_q = self._quantum_proj(q)
        v_q = self._quantum_proj(v)

        attn_out = self.attention(q_q, k_q, v_q, mask)
        return self.combine_heads(attn_out)


class FeedForwardBase(nn.Module):
    """Base class for feed‑forward networks."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class FeedForwardQuantum(FeedForwardBase):
    """Feed‑forward network realized by a variational quantum circuit."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int):
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
            for i, gate in enumerate(self.parameters):
                gate(q_device, wires=i)
            return self.measure(q_device)

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1):
        super().__init__(embed_dim, ffn_dim, dropout)
        self.n_qubits = n_qubits
        self.linear1 = nn.Linear(embed_dim, n_qubits, bias=False)
        self.linear2 = nn.Linear(n_qubits, embed_dim, bias=False)
        self.q_layer = self.QLayer(n_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        proj = self.linear1(x)  # (batch, seq_len, n_qubits)
        flat = proj.reshape(-1, self.n_qubits)
        qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=flat.shape[0], device=flat.device)
        out = self.q_layer(flat, qdev)
        out = out.reshape(batch, seq_len, self.n_qubits)
        out = self.linear2(out)
        return self.dropout(F.relu(out))


class TransformerBlockBase(nn.Module):
    """Base transformer block."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TransformerBlockQuantum(TransformerBlockBase):
    """Fully quantum transformer block."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 n_qubits_ffn: int, dropout: float = 0.1):
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class ProjectionHead(nn.Module):
    """MLP used for contrastive representation learning."""
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class TextClassifier(nn.Module):
    """Quantum‑enhanced transformer‑based text classifier."""
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        projection_dim: int = 0,
        n_qubits_ffn: int = 8,
        learn_positional: bool = False,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(embed_dim, learnable=learn_positional)
        self.blocks = nn.ModuleList(
            [TransformerBlockQuantum(embed_dim, num_heads, ffn_dim,
                                     n_qubits_ffn, dropout)
             for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(
            embed_dim, num_classes if num_classes > 2 else 1, bias=False
        )
        self.projection_dim = projection_dim
        self.proj_head = ProjectionHead(embed_dim, projection_dim) if projection_dim > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_encoder(tokens)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        logits = self.classifier(x)
        if self.proj_head is not None:
            proj = self.proj_head(x)
            return logits, proj
        return logits


__all__ = [
    "PositionalEncoding",
    "MultiHeadAttentionBase",
    "MultiHeadAttentionQuantum",
    "FeedForwardBase",
    "FeedForwardQuantum",
    "TransformerBlockBase",
    "TransformerBlockQuantum",
    "ProjectionHead",
    "TextClassifier",
]
