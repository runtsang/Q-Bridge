import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Classical utilities – unchanged from the seed
# --------------------------------------------------------------------------- #
class MultiHeadAttentionBase(nn.Module):
    """Base class for attention mechanisms."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.d_k = embed_dim // num_heads

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Standard multi‑head self‑attention."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq_len, _ = x.size()
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
        return self.out_proj(attn_output)


class MultiHeadAttentionHybrid(MultiHeadAttentionBase):
    """Hybrid attention that optionally applies quantum‑like per‑head transformations."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1,
                 use_quantum: bool = False, quantum_depth: int = 1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.use_quantum = use_quantum
        self.quantum_depth = quantum_depth
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        if self.use_quantum:
            # per‑head quantum‑like transforms
            self.q_heads = nn.ModuleList([
                nn.Sequential(*[nn.Linear(self.d_k, self.d_k) for _ in range(quantum_depth)])
                for _ in range(num_heads)
            ])
            self.k_heads = nn.ModuleList([
                nn.Sequential(*[nn.Linear(self.d_k, self.d_k) for _ in range(quantum_depth)])
                for _ in range(num_heads)
            ])
            self.v_heads = nn.ModuleList([
                nn.Sequential(*[nn.Linear(self.d_k, self.d_k) for _ in range(quantum_depth)])
                for _ in range(num_heads)
            ])

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq_len, _ = x.size()
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        if self.use_quantum:
            for i in range(self.num_heads):
                q[:, i] = self.q_heads[i](q[:, i])
                k[:, i] = self.k_heads[i](k[:, i])
                v[:, i] = self.v_heads[i](v[:, i])

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
        return self.out_proj(attn_output)


class FeedForwardBase(nn.Module):
    """Base class for feed‑forward networks."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)


class FeedForwardClassical(FeedForwardBase):
    """Two‑layer MLP."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encodings – unchanged."""
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(1, max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float) * (-math.log(10000.0) / embed_dim))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class TransformerBlockBase(nn.Module):
    """Base class with layer‑norm and dropout."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class TransformerBlockHybrid(nn.Module):
    """Hybrid transformer block using the hybrid attention."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 dropout: float = 0.1, use_quantum: bool = False,
                 quantum_depth: int = 1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn = MultiHeadAttentionHybrid(embed_dim, num_heads, dropout,
                                            use_quantum=use_quantum,
                                            quantum_depth=quantum_depth)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class VAEEncoder(nn.Module):
    """Variational auto‑encoder encoder used for feature extraction."""
    def __init__(self, embed_dim: int, latent_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, latent_dim)
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        eps = torch.randn_like(mu)
        z = mu + eps * torch.exp(0.5 * logvar)
        return z


class QuantumHybridTransformer(nn.Module):
    """Transformer‑based text classifier supporting hybrid quantum‑classical layers."""
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 latent_dim: Optional[int] = None,
                 use_vae: bool = False,
                 use_hybrid_attention: bool = False,
                 quantum_depth: int = 1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.use_vae = use_vae
        self.vae = VAEEncoder(embed_dim, latent_dim) if use_vae and latent_dim is not None else None
        self.use_hybrid_attention = use_hybrid_attention
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            if use_hybrid_attention:
                block = TransformerBlockHybrid(embed_dim, num_heads, ffn_dim,
                                              dropout=dropout,
                                              use_quantum=True,
                                              quantum_depth=quantum_depth)
            else:
                block = TransformerBlockBase(embed_dim, num_heads, ffn_dim,
                                            dropout=dropout)
            self.blocks.append(block)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)  # (batch, seq_len, embed_dim)
        if self.vae is not None:
            x = self.vae(x)
        x = self.pos_encoder(x)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=1)  # global average pooling
        x = self.dropout(x)
        return self.classifier(x)


__all__ = [
    "MultiHeadAttentionBase",
    "MultiHeadAttentionClassical",
    "MultiHeadAttentionHybrid",
    "FeedForwardBase",
    "FeedForwardClassical",
    "PositionalEncoder",
    "TransformerBlockBase",
    "TransformerBlockHybrid",
    "VAEEncoder",
    "QuantumHybridTransformer",
]
