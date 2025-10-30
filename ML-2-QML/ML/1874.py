import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class MultiHeadAttentionBase(nn.Module):
    """Base class for multi‑head attention."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1,
                 use_quantum: bool = False):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.use_quantum = use_quantum

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError

class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Standard multi‑head attention implemented with nn.MultiheadAttention."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1,
                 use_quantum: bool = False, **kwargs):
        super().__init__(embed_dim, num_heads, dropout, use_quantum)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.attn(x, x, x, key_padding_mask=mask)[0]

class MultiHeadAttentionQuantum(MultiHeadAttentionClassical):
    """Alias of the classical attention for API symmetry."""
    pass

class FeedForwardBase(nn.Module):
    """Base class for feed‑forward networks."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class FeedForwardClassical(FeedForwardBase):
    """Two‑layer perceptron feed‑forward network."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class FeedForwardQuantum(FeedForwardClassical):
    """Alias of the classical feed‑forward block."""
    pass

class TransformerBlockBase(nn.Module):
    """Base transformer block containing attention and feed‑forward parts."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class TransformerBlockClassical(TransformerBlockBase):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 dropout: float = 0.1):
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class TransformerBlockQuantum(TransformerBlockClassical):
    """Alias of the classical block for API symmetry."""
    pass

class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) *
                             (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class QuantumAttentionWrapper(nn.Module):
    """Wraps a classical attention output, then passes it through a simple
    quantum‑like non‑linearity."""
    def __init__(self, embed_dim: int, n_wires: int = 8):
        super().__init__()
        self.linear_to_wires = nn.Linear(embed_dim, n_wires)
        self.linear_back = nn.Linear(n_wires, embed_dim)
        self.n_wires = n_wires

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_w = self.linear_to_wires(x)
        x_q = torch.sin(x_w)  # quantum‑style measurement
        return self.linear_back(x_q)

class QuantumFeedForwardWrapper(nn.Module):
    """Wraps a classical feed‑forward block with a quantum‑style non‑linearity."""
    def __init__(self, embed_dim: int, ffn_dim: int, n_wires: int = 8):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.n_wires = n_wires

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x_q = torch.cos(x)  # quantum‑style non‑linearity
        return self.linear2(x_q)

class QuantumTransformerClassifier(nn.Module):
    """Hybrid transformer‑based text classifier supporting optional quantum sub‑modules."""
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 use_quantum_attention: bool = False,
                 use_quantum_ffn: bool = False,
                 quantum_wires: int = 8,
                 quantum_dropout_rate: float = 0.0):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.quantum_dropout = nn.Dropout(quantum_dropout_rate) if quantum_dropout_rate > 0 else nn.Identity()

        blocks = []
        for _ in range(num_blocks):
            # Attention
            if use_quantum_attention:
                attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
                attn = nn.Sequential(attn, QuantumAttentionWrapper(embed_dim, quantum_wires))
            else:
                attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)

            # Feed‑forward
            if use_quantum_ffn:
                ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)
                ffn = nn.Sequential(ffn, QuantumFeedForwardWrapper(embed_dim, ffn_dim, quantum_wires))
            else:
                ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

            block = TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
            if use_quantum_attention:
                block.attn = attn
            if use_quantum_ffn:
                block.ffn = ffn
            blocks.append(block)

        self.transformers = nn.Sequential(*blocks)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        x = self.transformers(x)
        x = x.mean(dim=1)
        x = self.quantum_dropout(x)
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
    "QuantumAttentionWrapper",
    "QuantumFeedForwardWrapper",
    "QuantumTransformerClassifier",
]
