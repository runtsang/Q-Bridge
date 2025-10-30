"""
HybridTransformer: a classical transformer with optional hybrid attention and efficient feed‑forward.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridAttention(nn.Module):
    """
    Hybrid multi‑head attention that mixes a classical linear projection with a quantum phase‑shift.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

        self.q_lin = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_lin = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_lin = nn.Linear(embed_dim, embed_dim, bias=False)

        # Quantum phase shift parameters: one per head per d_k dimension
        self.phase_shift = nn.Parameter(torch.randn(num_heads, self.d_k))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq_len, _ = x.size()
        q = self.q_lin(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_lin(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_lin(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Apply quantum phase shift: multiply by e^{i * phase}
        phase = self.phase_shift.unsqueeze(0).unsqueeze(2)  # shape (1, H, 1, d_k)
        q = q * torch.exp(1j * phase)
        k = k * torch.exp(1j * phase)
        v = v * torch.exp(1j * phase)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
        return out

class EfficientFFN(nn.Module):
    """
    Feed‑forward network with a single linear layer followed by a small quantum circuit.
    """
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlockHybrid(nn.Module):
    """
    Transformer block that uses HybridAttention and EfficientFFN.
    """
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = HybridAttention(embed_dim, num_heads, dropout)
        self.ffn = EfficientFFN(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class PositionalEncoder(nn.Module):
    """
    Sinusoidal positional encoding.
    """
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

class HybridTransformer(nn.Module):
    """
    Hybrid transformer that can operate in fully classical mode or with hybrid attention and efficient FFN.
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
        use_hybrid_attention: bool = True,
        use_efficient_ffn: bool = True,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            if use_hybrid_attention:
                attn = HybridAttention(embed_dim, num_heads, dropout)
            else:
                attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
            if use_efficient_ffn:
                ffn = EfficientFFN(embed_dim, ffn_dim, dropout)
            else:
                ffn = nn.Sequential(
                    nn.Linear(embed_dim, ffn_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(ffn_dim, embed_dim),
                )
            block = nn.ModuleDict({
                "attn": attn,
                "ffn": ffn,
                "norm1": nn.LayerNorm(embed_dim),
                "norm2": nn.LayerNorm(embed_dim),
                "dropout": nn.Dropout(dropout),
            })
            self.blocks.append(block)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        x = self.pos_embedding(x)
        for block in self.blocks:
            attn = block["attn"]
            ffn = block["ffn"]
            norm1 = block["norm1"]
            norm2 = block["norm2"]
            dropout = block["dropout"]

            if isinstance(attn, nn.MultiheadAttention):
                attn_out, _ = attn(x, x, x)
            else:
                attn_out = attn(x)
            x = norm1(x + dropout(attn_out))
            ffn_out = ffn(x)
            x = norm2(x + dropout(ffn_out))
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)

__all__ = ["HybridTransformer"]
