import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttentionBase(nn.Module):
    """Base class for multi‑head attention."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
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
        out, self.attn_weights = self.attention(q, k, v, mask)
        return out.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Standard multi‑head attention."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__(embed_dim, num_heads, dropout)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.combine = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq, _ = x.shape
        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)
        out = self.downstream(q, k, v, batch, mask)
        return self.combine(out)

class MultiHeadAttentionHybrid(MultiHeadAttentionBase):
    """Hybrid attention mixing classical and a lightweight quantum‑like branch."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, quantum_weight: float = 0.5):
        super().__init__(embed_dim, num_heads, dropout)
        self.quantum_weight = nn.Parameter(torch.tensor(quantum_weight))
        self.classical = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.quantum_heads = nn.ModuleList([nn.Linear(embed_dim // num_heads, embed_dim // num_heads) for _ in range(num_heads)])

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        class_out = self.classical(x, mask)
        q_out_parts = []
        batch, seq, _ = x.shape
        for head_idx in range(self.num_heads):
            head_x = x[:, :, head_idx * self.d_k:(head_idx + 1) * self.d_k]
            transformed = self.quantum_heads[head_idx](head_x)
            q_out_parts.append(transformed)
        q_out = torch.cat(q_out_parts, dim=2)
        return self.quantum_weight * q_out + (1 - self.quantum_weight) * class_out

class FeedForwardBase(nn.Module):
    """Base feed‑forward network."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

class FeedForwardClassical(FeedForwardBase):
    """Two‑layer MLP."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class FeedForwardHybrid(FeedForwardBase):
    """Hybrid feed‑forward: classical MLP on top of a quantum‑like embedding."""
    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int = 4, dropout: float = 0.1):
        super().__init__(embed_dim, ffn_dim, dropout)
        self.n_qubits = n_qubits
        self.encoder = nn.Linear(embed_dim, n_qubits)
        self.circuit = nn.Linear(n_qubits, n_qubits)
        self.classical = nn.Linear(n_qubits, ffn_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.encoder(x)
        q = self.circuit(q)
        q = self.dropout(q)
        return self.classical(q)

class TransformerBlockBase(nn.Module):
    """Base transformer block with pre‑norm."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

class TransformerBlockHybrid(nn.Module):
    """Hybrid transformer block mixing attention and feed‑forward."""
    def __init__(self, embed_dim: int, num_heads: int,
                 ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadAttentionHybrid(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardHybrid(embed_dim, ffn_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class PositionalEncoderLearnable(nn.Module):
    """Learnable positional encoding with optional dropout."""
    def __init__(self, embed_dim: int, max_len: int = 5000, dropout: float = 0.0):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, embed_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = self.pe[:, :x.size(1)]
        return x + self.dropout(pos)

class TextClassifier(nn.Module):
    """Hybrid transformer‑based text classifier with optional quantum‑like mixing."""
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 use_hybrid: bool = True,
                 quantum_weight: float = 0.5,
                 pos_dropout: float = 0.0):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoderLearnable(embed_dim, dropout=pos_dropout)
        if use_hybrid:
            block_cls = TransformerBlockHybrid
        else:
            block_cls = lambda embed_dim, num_heads, ffn_dim, dropout: TransformerBlockBase(embed_dim, num_heads, dropout)
        self.transformers = nn.Sequential(*[block_cls(embed_dim, num_heads, ffn_dim, dropout)
                                            for _ in range(num_blocks)])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        x = self.pos_encoder(x)
        x = self.transformers(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)

__all__ = [
    "MultiHeadAttentionBase",
    "MultiHeadAttentionClassical",
    "MultiHeadAttentionHybrid",
    "FeedForwardBase",
    "FeedForwardClassical",
    "FeedForwardHybrid",
    "TransformerBlockBase",
    "TransformerBlockHybrid",
    "PositionalEncoderLearnable",
    "TextClassifier",
]
