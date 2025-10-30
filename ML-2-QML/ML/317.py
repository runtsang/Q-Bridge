import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttentionBase(nn.Module):
    """Base class for all attention variants.  Keeps the embed‑size checks and
    the helper that splits the tensor into heads."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.d_k = embed_dim // num_heads

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


class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Pure‑classical multi‑head attention using PyTorch's MultiheadAttention."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.attn(x, x, x, key_padding_mask=mask)[0]


class QuantumAttentionHead(nn.Module):
    """Single‑head quantum attention that maps a token through a small circuit."""
    def __init__(self, n_wires: int, device: Optional[torch.device] = None):
        super().__init__()
        self.n_wires = n_wires
        self.qdevice = torch.quantum.QDevice(n_wires=n_wires, device=device or 'cpu')
        self.encoder = torch.quantum.general_encoder(
            [{'input_idx': [i], 'func': 'rx', 'wires': [i]} for i in range(n_wires)])
        self.param_gates = nn.ModuleList(
            [nn.Parameter(torch.randn(1)) for _ in range(n_wires)])
        self.measure = torch.quantum.MeasureAll(tq.PauliZ)

    def forward(self, token: torch.Tensor) -> torch.Tensor:
        self.encoder(self.qdevice, token)
        for i, gate in enumerate(self.param_gates):
            gate(self.qdevice, wires=i)
        return self.measure(self.qdevice)


class MultiHeadAttentionHybrid(MultiHeadAttentionBase):
    """Hybrid attention that uses classical projections for keys/values and a quantum head for queries."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1,
                 n_wires: int = 4):
        super().__init__(embed_dim, num_heads, dropout)
        self.q_head = QuantumAttentionHead(n_wires)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.combine = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq, _ = x.shape
        q = torch.stack([self.q_head(x[:, i, :]) for i in range(seq)], dim=1)
        k = self.k_linear(x)
        v = self.v_linear(x)
        q = self.separate_heads(q)
        k = self.separate_heads(k)
        v = self.separate_heads(v)
        attn_scores, _ = self.attention(q, k, v, mask)
        attn_output = torch.matmul(attn_scores, v)
        attn_output = self.restore_heads(attn_output)
        return self.combine(attn_output)


class FeedForwardBase(nn.Module):
    """Base class for feed‑forward layers."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class FeedForwardClassical(FeedForwardBase):
    """Standard two‑layer MLP."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class FeedForwardQuantum(FeedForwardBase):
    """Feed‑forward realized by a quantum module."""
    class QLayer(nn.Module):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = torch.quantum.general_encoder(
                [{'input_idx': [i], 'func': 'ry', 'wires': [i]} for i in range(n_wires)])
            self.params = nn.ModuleList(
                [nn.Parameter(torch.randn(1)) for _ in range(n_wires)])
            self.measure = torch.quantum.MeasureAll(tq.PauliZ)

        def forward(self, token: torch.Tensor) -> torch.Tensor:
            self.encoder(self.qdevice, token)
            for i, gate in enumerate(self.params):
                gate(self.qdevice, wires=i)
            return self.measure(self.qdevice)

    def __init__(self, embed_dim: int, ffn_dim: int, n_wires: int, dropout: float = 0.1):
        super().__init__(embed_dim, ffn_dim, dropout)
        self.qlayer = self.QLayer(n_wires)
        self.qdevice = torch.quantum.QDevice(n_wires=n_wires)
        self.linear1 = nn.Linear(n_wires, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for token in x.unbind(dim=1):
            out = self.qlayer(token)
            outputs.append(out)
        out = torch.stack(outputs, dim=1)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))


class TransformerBlockBase(nn.Module):
    """Base block containing attention and feed‑forward."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TransformerBlockHybrid(TransformerBlockBase):
    """Transformer block that can be classical, quantum, or hybrid."""
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 use_quantum_attn: bool = False,
                 use_quantum_ffn: bool = False,
                 n_wires_attn: int = 4,
                 n_wires_ffn: int = 4,
                 dropout: float = 0.1):
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = (MultiHeadAttentionQuantum(embed_dim, num_heads, dropout, n_wires_attn)
                     if use_quantum_attn else MultiHeadAttentionHybrid(embed_dim, num_heads, dropout, n_wires_attn))
        self.ffn = (FeedForwardQuantum(embed_dim, ffn_dim, n_wires_ffn, dropout)
                    if use_quantum_ffn else FeedForwardClassical(embed_dim, ffn_dim, dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding with optional learnable phase shift."""
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


class TextClassifier(nn.Module):
    """Transformer‑based text classifier supporting hybrid quantum submodules."""
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 use_quantum_attn: bool = False,
                 use_quantum_ffn: bool = False,
                 n_wires_attn: int = 4,
                 n_wires_ffn: int = 4,
                 learnable_pos: bool = False):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim, learnable_phase=learnable_pos)
        self.blocks = nn.ModuleList(
            [TransformerBlockHybrid(embed_dim,
                                    num_heads,
                                    ffn_dim,
                                    use_quantum_attn,
                                    use_quantum_ffn,
                                    n_wires_attn,
                                    n_wires_ffn,
                                    dropout)
             for _ in range(num_blocks)])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        x = self.pos_encoder(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)


__all__ = [
    "MultiHeadAttentionBase",
    "MultiHeadAttentionClassical",
    "MultiHeadAttentionHybrid",
    "FeedForwardBase",
    "FeedForwardClassical",
    "FeedForwardQuantum",
    "TransformerBlockBase",
    "TransformerBlockHybrid",
    "PositionalEncoder",
    "TextClassifier",
]
