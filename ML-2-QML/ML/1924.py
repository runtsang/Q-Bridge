"""Enhanced classical convolution‑inspired network with multi‑head attention."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class AttentionBlock(nn.Module):
    """Single transformer‑style block with residuals, layer norm and feed‑forward."""
    def __init__(self, hidden_dim: int, heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_norm = nn.LayerNorm(hidden_dim)

        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.ff_dropout = nn.Dropout(dropout)
        self.ff_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multi‑head self‑attention
        attn_out, _ = self.attn(x, x, x)
        x = x + self.attn_dropout(attn_out)
        x = self.attn_norm(x)

        # Feed‑forward sub‑layer
        ff_out = self.ff(x)
        x = x + self.ff_dropout(ff_out)
        x = self.ff_norm(x)
        return x


class QCNNEncoder(nn.Module):
    """
    Classic encoder that mimics a QCNN via a sequence of linear layers
    and transformer‑style attention blocks. The design mirrors the
    original seed but adds positional encoding, residual connections,
    and attention for deeper representation learning.
    """
    def __init__(
        self,
        in_features: int = 8,
        hidden_dim: int = 16,
        heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim

        # Token projection: map each scalar feature to a hidden vector
        self.token_proj = nn.Linear(1, hidden_dim)

        # Learnable positional encoding
        self.positional = nn.Parameter(torch.randn(in_features, hidden_dim))

        # Stack of attention blocks
        self.layers = nn.ModuleList(
            [AttentionBlock(hidden_dim, heads, dropout) for _ in range(num_layers)]
        )

        # Final projection to the 4‑dimensional feature space used by the head
        self.projection = nn.Linear(hidden_dim, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Shape (batch_size, in_features)

        Returns:
            torch.Tensor: Shape (batch_size, 4)
        """
        # (batch, seq_len, 1)
        x = x.unsqueeze(-1)

        # (batch, seq_len, hidden_dim)
        x = self.token_proj(x)

        # Add positional encoding
        x = x + self.positional.unsqueeze(0)

        # Pass through attention layers
        for layer in self.layers:
            x = layer(x)

        # Global average pooling over the sequence dimension
        x = x.mean(dim=1)

        # Project to 4‑dimensional feature vector
        return self.projection(x)


class QCNNEnhanced(nn.Module):
    """Full hybrid QCNN model: attention encoder + classification head."""
    def __init__(
        self,
        in_features: int = 8,
        hidden_dim: int = 16,
        heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = QCNNEncoder(
            in_features=in_features,
            hidden_dim=hidden_dim,
            heads=heads,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        return torch.sigmoid(self.head(x))


def QCNNEnhancedFactory() -> QCNNEnhanced:
    """
    Factory that returns a ready‑to‑train :class:`QCNNEnhanced` instance.
    """
    return QCNNEnhanced()


__all__ = ["QCNNEnhanced", "QCNNEnhancedFactory"]
