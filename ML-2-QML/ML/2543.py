"""Unified estimator that blends a simple regressor with a transformer backbone."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# Simple regressor
class _SimpleRegressor(nn.Module):
    """Minimal fully‑connected network mirroring EstimatorQNN."""
    def __init__(self, in_features: int = 2, hidden: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# Classical transformer components
class MultiHeadAttentionClassical(nn.Module):
    """Standard multi‑head attention implemented with torch.nn.MultiheadAttention."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_output, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_output

class FeedForwardClassical(nn.Module):
    """Two‑layer perceptron feed‑forward network."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlockClassical(nn.Module):
    """Transformer block using classical attention and feed‑forward."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]

# Quantum placeholders (alias to classical for compatibility)
MultiHeadAttentionQuantum = MultiHeadAttentionClassical
FeedForwardQuantum = FeedForwardClassical
TransformerBlockQuantum = TransformerBlockClassical

# Hybrid transformer backbone
class _TransformerBackbone(nn.Module):
    """Hybrid transformer that can swap classical or quantum modules."""
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        *,
        use_quantum_attn: bool = False,
        use_quantum_ffn: bool = False,
        q_device: Optional[object] = None,
    ) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_enc = PositionalEncoder(embed_dim)
        AttnCls = MultiHeadAttentionQuantum if use_quantum_attn else MultiHeadAttentionClassical
        FfnCls = FeedForwardQuantum if use_quantum_ffn else FeedForwardClassical
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(
                TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout=0.1)
            )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_emb(x)
        x = self.pos_enc(x)
        for block in self.blocks:
            x = block(x)
        return x.mean(dim=1)

# Unified estimator
class UnifiedEstimatorTransformer(nn.Module):
    """High‑level estimator that exposes both regression and transformer modes."""
    def __init__(
        self,
        mode: str = "regression",
        *,
        # regression params
        in_features: int = 2,
        hidden: int = 8,
        # transformer params
        vocab_size: int = 1000,
        embed_dim: int = 32,
        num_heads: int = 4,
        num_blocks: int = 2,
        ffn_dim: int = 64,
        use_quantum_attn: bool = False,
        use_quantum_ffn: bool = False,
        q_device: Optional[object] = None,
    ) -> None:
        super().__init__()
        if mode == "regression":
            self.model = _SimpleRegressor(in_features, hidden)
        elif mode == "transformer":
            self.model = _TransformerBackbone(
                vocab_size,
                embed_dim,
                num_heads,
                num_blocks,
                ffn_dim,
                use_quantum_attn=use_quantum_attn,
                use_quantum_ffn=use_quantum_ffn,
                q_device=q_device,
            )
        else:
            raise ValueError(f"Unsupported mode {mode}")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

def EstimatorQNN(**kwargs) -> UnifiedEstimatorTransformer:
    """Convenience wrapper that matches the original EstimatorQNN API."""
    return UnifiedEstimatorTransformer(**kwargs)
