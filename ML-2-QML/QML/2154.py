"""
TextClassifier – Hybrid quantum–classical transformer.

This module re‑exports the same public classes as the classical file but
adds optional quantum sub‑modules.  The quantum path uses TorchQuantum
to encode the attention heads and the feed‑forward inner dimension into
parameterised qubit circuits.  A single reusable quantum device is
created per model instance to avoid per‑token overhead.  The `use_q`
flag toggles the hybrid behaviour and keeps the original API
backwards compatible.
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
#  Base abstractions – identical to classical seed
# --------------------------------------------------------------------------- #
class MultiHeadAttentionBase(nn.Module):
    """Base class for all attention variants."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1,
                 mask: Optional[torch.Tensor] = None, use_bias: bool = False) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.attn_weights: Optional[torch.Tensor] = None

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        return x.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                  id_mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if id_mask is not None:
            scores = scores.masked_fill(id_mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        self.attn_weights = scores
        return torch.matmul(scores, v), scores

    def downstream(self, **kwargs):
        raise NotImplementedError


# --------------------------------------------------------------------------- #
#  Classical attention – unchanged
# --------------------------------------------------------------------------- #
class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1,
                 mask: Optional[torch.Tensor] = None, use_bias: bool = False) -> None:
        super().__init__(embed_dim, num_heads, dropout, mask, use_bias)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.combine_heads = nn.Linear(embed_dim, embed_dim, bias=use_bias)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)
        q = self.separate_heads(q)
        k = self.separate_heads(k)
        v = self.separate_heads(v)
        out, _ = self.attention(q, k, v, id_mask=mask)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.combine_heads(out)


# --------------------------------------------------------------------------- #
#  Quantum attention – new component
# --------------------------------------------------------------------------- #
class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    """Attention where the key, query and value projections are processed
    through a small quantum circuit per head.  A single reusable quantum
    device is reused across all tokens to avoid device‑creation overhead.
    """
    class _QuantumHead(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            # Encode vector into qubits
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            # Parameterised rotation gates
            self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev, x)
            for gate, wire in zip(self.params, range(self.n_wires)):
                gate(qdev, wires=[wire])
            return self.measure(qdev)

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1,
                 mask: Optional[torch.Tensor] = None, use_bias: bool = False,
                 n_wires: int = 8, q_device: Optional[tq.QuantumDevice] = None) -> None:
        super().__init__(embed_dim, num_heads, dropout, mask, use_bias)
        self.n_wires = n_wires
        self.q_head = self._QuantumHead(n_wires)
        self.q_device = q_device or tq.QuantumDevice(n_wires=n_wires)
        self.combine_heads = nn.Linear(embed_dim, embed_dim, bias=use_bias)

    def _quantum_proj(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the quantum circuit to each projection head."""
        batch, seq, heads, dim = x.shape  # heads = num_heads
        # reshape to (batch*seq*heads, dim)
        flat = x.reshape(-1, dim)
        outputs = []
        bsz = flat.shape[0]
        for token in flat.unbind(dim=0):
            qdev = self.q_device.copy(bsz=1, device=token.device)
            outputs.append(self.q_head(token.unsqueeze(0), qdev))
        out = torch.stack(outputs, dim=0).view(batch, seq, heads, dim)
        return out

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Linear projections
        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)
        # reshape for heads
        k = self.separate_heads(k)
        q = self.separate_heads(q)
        v = self.separate_heads(v)
        # quantum processing per head
        k = self._quantum_proj(k)
        q = self._quantum_proj(q)
        v = self._quantum_proj(v)
        out, _ = self.attention(q, k, v, id_mask=mask)
        out = out.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], self.embed_dim)
        return self.combine_heads(out)


# --------------------------------------------------------------------------- #
#  Feed‑forward – quantum variant
# --------------------------------------------------------------------------- #
class FeedForwardQuantum(FeedForwardBase):
    """Feed‑forward layer realised by a small quantum circuit."""
    class _QuantumFF(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.params = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev, x)
            for gate, wire in zip(self.params, range(self.n_wires)):
                gate(qdev, wires=[wire])
            return self.measure(qdev)

    def __init__(self, embed_dim: int, ffn_dim: int, n_wires: int = 8, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.q_ff = self._QuantumFF(n_wires)
        self.q_device = tq.QuantumDevice(n_wires=n_wires)
        self.linear1 = nn.Linear(n_wires, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # apply quantum circuit to each token
        batch, seq, _ = x.shape
        out = []
        for token in x.unbind(dim=1):
            qdev = self.q_device.copy(bsz=token.size(0), device=token.device)
            out.append(self.q_ff(token, qdev))
        out = torch.stack(out, dim=1)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))


# --------------------------------------------------------------------------- #
#  Transformer block – quantum variant
# --------------------------------------------------------------------------- #
class TransformerBlockQuantum(TransformerBlockBase):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 n_wires_attn: int = 8,
                 n_wires_ffn: int = 8,
                 dropout: float = 0.1,
                 use_q: bool = True) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = (MultiHeadAttentionQuantum(embed_dim, num_heads, dropout,
                                               n_wires=n_wires_attn) if use_q
                     else MultiHeadAttentionClassical(embed_dim, num_heads, dropout))
        self.ffn = (FeedForwardQuantum(embed_dim, ffn_dim, n_wires=n_wires_ffn) if use_q
                    else FeedForwardClassical(embed_dim, ffn_dim, dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


# --------------------------------------------------------------------------- #
#  Positional encoding – same as classical
# --------------------------------------------------------------------------- #
class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""
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


# --------------------------------------------------------------------------- #
#  Text classifier – hybrid
# --------------------------------------------------------------------------- #
class TextClassifier(nn.Module):
    """Transformer‑based text classifier with optional quantum sub‑modules."""
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 use_q: bool = False,
                 n_wires_attn: int = 8,
                 n_wires_ffn: int = 8) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        block_cls = (TransformerBlockQuantum if use_q else TransformerBlockClassical)
        self.transformers = nn.Sequential(
            *[block_cls(embed_dim, num_heads, ffn_dim,
                        n_wires_attn=n_wires_attn,
                        n_wires_ffn=n_wires_ffn,
                        dropout=dropout,
                        use_q=use_q)
              for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
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
