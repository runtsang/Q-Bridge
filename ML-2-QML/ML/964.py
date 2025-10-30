import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttentionBase(nn.Module):
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

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError

class MultiHeadAttentionHybrid(MultiHeadAttentionBase):
    """Hybrid attention that can switch between classical linear projection and a quantum‑encoded projection."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, use_quantum: bool = False):
        super().__init__(embed_dim, num_heads, dropout)
        self.use_quantum = use_quantum
        # Classical linear projections
        self.k_linear_cls = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_linear_cls = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear_cls = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj_cls = nn.Linear(embed_dim, embed_dim, bias=False)
        # Quantum‑like linear projections (placeholder)
        self.k_linear_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_linear_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj_q = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_quantum:
            k = self.k_linear_q(x)
            q = self.q_linear_q(x)
            v = self.v_linear_q(x)
            out_proj = self.out_proj_q
        else:
            k = self.k_linear_cls(x)
            q = self.q_linear_cls(x)
            v = self.v_linear_cls(x)
            out_proj = self.out_proj_cls

        k = self.separate_heads(k)
        q = self.separate_heads(q)
        v = self.separate_heads(v)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2), -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], self.embed_dim)
        return out_proj(out)

class FeedForwardBase(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class FeedForwardHybrid(FeedForwardBase):
    """Hybrid feed‑forward that can switch between classical and a quantum‑like linear path."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1, use_quantum: bool = False):
        super().__init__(embed_dim, ffn_dim, dropout)
        self.use_quantum = use_quantum
        # Classical path
        self.linear1_cls = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.linear2_cls = nn.Linear(ffn_dim, embed_dim, bias=False)
        # Quantum‑like path (placeholder)
        self.linear1_q = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.linear2_q = nn.Linear(ffn_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_quantum:
            out = self.linear2_q(F.relu(self.linear1_q(x)))
        else:
            out = self.linear2_cls(F.relu(self.linear1_cls(x)))
        return self.dropout(out)

class TransformerBlockBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class TransformerBlockHybrid(TransformerBlockBase):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1,
                 use_quantum_attn: bool = False, use_quantum_ffn: bool = False):
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionHybrid(embed_dim, num_heads, dropout, use_quantum=use_quantum_attn)
        self.ffn = FeedForwardHybrid(embed_dim, ffn_dim, dropout, use_quantum=use_quantum_ffn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class PositionalEmbedding(nn.Module):
    """Learnable positional embedding."""
    def __init__(self, max_len: int, embed_dim: int):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pos_embedding[:, :seq_len, :]

class TextClassifier(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 max_len: int = 512,
                 use_quantum_attn: bool = False,
                 use_quantum_ffn: bool = False):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEmbedding(max_len, embed_dim)
        self.transformers = nn.Sequential(
            *[TransformerBlockHybrid(embed_dim, num_heads, ffn_dim, dropout,
                                     use_quantum_attn=use_quantum_attn,
                                     use_quantum_ffn=use_quantum_ffn)
              for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        if num_classes > 2:
            self.classifier = nn.Linear(embed_dim, num_classes)
        else:
            self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        x = self.pos_embedding(x)
        x = self.transformers(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)

__all__ = [
    "MultiHeadAttentionBase",
    "MultiHeadAttentionHybrid",
    "FeedForwardBase",
    "FeedForwardHybrid",
    "TransformerBlockBase",
    "TransformerBlockHybrid",
    "PositionalEmbedding",
    "TextClassifier",
]
