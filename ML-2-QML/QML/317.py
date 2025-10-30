import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class MultiHeadAttentionBase(tq.QuantumModule):
    """Base class for quantum‑aware attention, keeps the embedding checks."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        return x.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)

    def restore_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, heads, seq, dim = x.shape
        return x.transpose(1, 2).contiguous().view(batch, seq, self.embed_dim)

    def attention(self,
                   query: torch.Tensor,
                   key: torch.Tensor,
                   value: torch.Tensor,
                   mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1), -1e9)
        return F.softmax(scores, dim=-1), scores

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError


class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    """Quantum‑aware attention that applies a small variational circuit to each head."""
    class QHead(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)])
            self.params = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, token: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, token)
            for wire, gate in enumerate(self.params):
                gate(q_device, wires=wire)
            return self.measure(q_device)

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.1,
                 n_wires: int = 4,
                 q_device: Optional[tq.QuantumDevice] = None):
        super().__init__(embed_dim, num_heads, dropout)
        self.qhead = self.QHead(n_wires)
        self.q_device = q_device or tq.QuantumDevice(n_wires=n_wires)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.combine = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq, _ = x.shape
        q = torch.stack([self.qhead(token, self.q_device) for token in x.unbind(dim=1)], dim=1)
        k = self.k_linear(x)
        v = self.v_linear(x)
        q = self.separate_heads(q)
        k = self.separate_heads(k)
        v = self.separate_heads(v)
        attn_scores, _ = self.attention(q, k, v, mask)
        attn_out = torch.matmul(attn_scores, v)
        attn_out = self.restore_heads(attn_out)
        return self.combine(attn_out)


class FeedForwardBase(tq.QuantumModule):
    """Base class for feed‑forward, supports classical and quantum variants."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class FeedForwardQuantum(tq.QuantumModule):
    """Feed‑forward realized by a variational circuit."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_wires)])
            self.params = nn.ModuleList(
                [tq.RY(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, token: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, token)
            for wire, gate in enumerate(self.params):
                gate(q_device, wires=wire)
            return self.measure(q_device)

    def __init__(self,
                 embed_dim: int,
                 ffn_dim: int,
                 n_wires: int,
                 dropout: float = 0.1,
                 q_device: Optional[tq.QuantumDevice] = None):
        super().__init__(embed_dim, ffn_dim, dropout)
        self.q_layer = self.QLayer(n_wires)
        self.q_device = q_device or tq.QuantumDevice(n_wires=n_wires)
        self.linear1 = nn.Linear(n_wires, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for token in x.unbind(dim=1):
            out = self.q_layer(token, self.q_device)
            outputs.append(out)
        out = torch.stack(outputs, dim=1)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))


class TransformerBlockQuantum(tq.QuantumModule):
    """Quantum‑enabled transformer block that can use quantum attention or feed‑forward."""
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 n_wires_attn: int,
                 n_wires_ffn: int,
                 dropout: float = 0.1,
                 use_q_attn: bool = True,
                 use_q_ffn: bool = True,
                 q_device: Optional[tq.QuantumDevice] = None):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn = (MultiHeadAttentionQuantum(embed_dim,
                                               num_heads,
                                               dropout,
                                               n_wires_attn,
                                               q_device)
                     if use_q_attn else
                     MultiHeadAttentionBase(embed_dim, num_heads, dropout))
        self.ffn = (FeedForwardQuantum(embed_dim,
                                       ffn_dim,
                                       n_wires_ffn,
                                       dropout,
                                       q_device)
                    if use_q_ffn else
                    FeedForwardBase(embed_dim, ffn_dim, dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class PositionalEncoder(tq.QuantumModule):
    """Sinusoidal positional encoder with optional learnable phase."""
    def __init__(self, embed_dim: int, max_len: int = 5000, learnable_phase: bool = False):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
        self.learnable_phase = learnable_phase
        if learnable_phase:
            self.phase = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        pe = self.pe[:, :seq_len]
        if self.learnable_phase:
            pe = pe * self.phase
        return x + pe


class TextClassifierQuantum(tq.QuantumModule):
    """Quantum‑enabled text classifier using hybrid transformer blocks."""
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 n_wires_attn: int = 4,
                 n_wires_ffn: int = 4,
                 use_q_attn: bool = True,
                 use_q_ffn: bool = True,
                 learnable_pos: bool = False,
                 q_device: Optional[tq.QuantumDevice] = None):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim,
                                             learnable_phase=learnable_pos)
        self.blocks = nn.ModuleList(
            [TransformerBlockQuantum(embed_dim,
                                     num_heads,
                                     ffn_dim,
                                     n_wires_attn,
                                     n_wires_ffn,
                                     dropout,
                                     use_q_attn,
                                     use_q_ffn,
                                     q_device)
             for _ in range(num_blocks)])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        x = self.pos_encoder(x)
        for blk in self.blocks:
            x = blk(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)


__all__ = [
    "MultiHeadAttentionBase",
    "MultiHeadAttentionQuantum",
    "FeedForwardBase",
    "FeedForwardQuantum",
    "TransformerBlockQuantum",
    "PositionalEncoder",
    "TextClassifierQuantum",
]
