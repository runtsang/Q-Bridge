"""Hybrid classical sampler network combining LSTM, Transformer, and kernel modules."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Positional encoding ------------------------------------
class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding for sequence data."""
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

# --- Classical transformer block ------------------------------------
class TransformerBlockClassical(nn.Module):
    """Standard transformer block with multi‑head attention and feed‑forward."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

# --- Classical RBF kernel ------------------------------------
class Kernel(nn.Module):
    """Radial basis function kernel used for similarity weighting."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

# --- Classical LSTM tagger ------------------------------------
class LSTMTagger(nn.Module):
    """Sequence tagging model using a classical LSTM."""
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embeddings(x)
        lstm_out, _ = self.lstm(emb)
        logits = self.classifier(lstm_out)
        return logits.squeeze(-1)

# --- Hybrid sampler network ------------------------------------
class SamplerQNN(nn.Module):
    """Hybrid sampler that processes sequences with LSTM, transformer, and kernel weighting."""
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        gamma: float = 1.0,
        reference_vector: torch.Tensor | None = None
    ) -> None:
        super().__init__()
        self.lstm_tagger = LSTMTagger(vocab_size, embed_dim, embed_dim)
        self.transformer = nn.Sequential(
            *[TransformerBlockClassical(embed_dim, num_heads, ffn_dim) for _ in range(num_blocks)]
        )
        self.kernel = Kernel(gamma)
        self.positional = PositionalEncoder(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)
        self.reference_vector = reference_vector or torch.randn(1, embed_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len)
        lstm_out, _ = self.lstm_tagger(x)
        # Add positional encoding
        x = self.positional(lstm_out)
        # Transformer processing
        x = self.transformer(x)
        # Global pooling
        x = x.mean(dim=1)
        # Kernel weighting
        sim = self.kernel(x, self.reference_vector)
        x = x * sim
        logits = self.classifier(x)
        return F.softmax(logits, dim=-1)

__all__ = ["SamplerQNN"]
