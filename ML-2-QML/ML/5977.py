import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttentionBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.d_k = embed_dim // num_heads

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, dim = x.shape
        return x.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, heads, seq, d_k = x.shape
        return x.transpose(1, 2).contiguous().view(batch, seq, self.embed_dim)

class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        k = self.k_proj(x)
        q = self.q_proj(x)
        v = self.v_proj(x)
        qk = self.separate_heads(q)
        k = self.separate_heads(k)
        v = self.separate_heads(v)
        scores = torch.matmul(qk, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores.masked_fill_(mask.unsqueeze(1), -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        return self.combine_heads(self.out_proj(out))

class MultiHeadAttentionHybrid(MultiHeadAttentionBase):
    def __init__(self, embed_dim: int, num_heads: int,
                 quantum_module: nn.Module,
                 dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.quantum_module = quantum_module
        self.classical_proj = nn.Linear(embed_dim, embed_dim)
        self.gate = nn.Parameter(torch.ones(num_heads) * 0.5)

    def _quantum_attention(self, x: torch.Tensor) -> torch.Tensor:
        return self.quantum_module(x)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        classical = self.classical_proj(x)
        quantum = self._quantum_attention(x)
        gate = self.gate.softmax(dim=0).unsqueeze(0).unsqueeze(2)
        hybrid = gate * quantum + (1 - gate) * classical
        return hybrid

class FeedForwardBase(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

class FeedForwardClassical(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, token: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(token))))

class FeedForwardQuantum(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, n_qbits: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_qbits = n_qbits
        self.linear1 = nn.Linear(n_qbits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qubits = x[:, :, :self.n_qbits]
        out = self.linear1(self.dropout(qubits))
        return self.linear2(F.relu(out))

class TransformerBlockBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

class TransformerBlockClassical(TransformerBlockBase):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class TransformerBlockHybrid(TransformerBlockBase):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 quantum_attn_module: nn.Module,
                 quantum_ffn_module: Optional[nn.Module] = None,
                 dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionHybrid(embed_dim, num_heads,
                                             quantum_module=quantum_attn_module,
                                             dropout=dropout)
        if quantum_ffn_module:
            self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qbits=4, dropout=dropout)
        else:
            self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000,
                 dropout: float = 0.1) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) *
                             (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class HybridTransformerClassifier(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 mode: str = "classical",
                 quantum_attn_module: Optional[nn.Module] = None,
                 quantum_ffn_module: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim, dropout=dropout)
        if mode == "classical":
            blocks = [TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
                      for _ in range(num_blocks)]
        elif mode == "hybrid":
            if quantum_attn_module is None:
                raise ValueError("Hybrid mode requires a quantum attention module")
            blocks = [TransformerBlockHybrid(embed_dim, num_heads, ffn_dim,
                                            quantum_attn_module,
                                            quantum_ffn_module,
                                            dropout) for _ in range(num_blocks)]
        else:  # quantum
            if quantum_attn_module is None:
                raise ValueError("Quantum mode requires a quantum attention module")
            blocks = [TransformerBlockHybrid(embed_dim, num_heads, ffn_dim,
                                            quantum_attn_module,
                                            quantum_ffn_module,
                                            dropout) for _ in range(num_blocks)]
        self.transformers = nn.Sequential(*blocks)
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
    "MultiHeadAttentionHybrid",
    "FeedForwardBase",
    "FeedForwardClassical",
    "FeedForwardQuantum",
    "TransformerBlockBase",
    "TransformerBlockClassical",
    "TransformerBlockHybrid",
    "PositionalEncoder",
    "HybridTransformerClassifier",
]
