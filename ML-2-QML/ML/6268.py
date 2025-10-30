"""QuantumHybridClassifier – classical implementation.

The module defines a hybrid classifier that can operate as a
two‑layer MLP or a transformer‑based model.  The architecture
mirrors the original QuantumClassifierModel but is fully classical
and uses PyTorch.  The class can be instantiated with a ``use_transformer``
flag to select the backbone.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumHybridClassifier(nn.Module):
    """Hybrid classifier with optional transformer backbone.

    Parameters
    ----------
    input_dim : int
        Dimension of the input feature vector or vocab size.
    hidden_dim : int
        Size of hidden layers (used by MLP) or embedding dimension (used by transformer).
    depth : int
        Number of layers in the MLP; ignored when using transformer.
    use_transformer : bool
        If True, the model uses a transformer encoder; otherwise it uses an MLP.
    num_heads : int
        Number of attention heads (only used when ``use_transformer=True``).
    num_blocks : int
        Number of transformer blocks.
    ffn_dim : int
        Feed‑forward dimension inside transformer blocks.
    num_classes : int
        Number of target classes.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        depth: int = 1,
        use_transformer: bool = False,
        num_heads: int = 4,
        num_blocks: int = 2,
        ffn_dim: int = 64,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.use_transformer = use_transformer
        self.num_classes = num_classes

        if use_transformer:
            # Transformer backbone
            self.token_embedding = nn.Embedding(input_dim, hidden_dim)
            self.pos_encoder = PositionalEncoder(hidden_dim)
            self.transformer = nn.Sequential(
                *[TransformerBlockClassical(hidden_dim, num_heads, ffn_dim)
                  for _ in range(num_blocks)]
            )
            self.classifier = nn.Linear(hidden_dim, num_classes)
        else:
            # MLP backbone
            layers = []
            in_dim = input_dim
            for _ in range(depth):
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.ReLU())
                in_dim = hidden_dim
            layers.append(nn.Linear(hidden_dim, num_classes))
            self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_transformer:
            x = self.token_embedding(x)
            x = self.pos_encoder(x)
            x = self.transformer(x)
            x = x.mean(dim=1)
            return self.classifier(x)
        else:
            return self.model(x)

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

class TransformerBlockClassical(nn.Module):
    """Purely classical transformer block."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

__all__ = ["QuantumHybridClassifier", "PositionalEncoder", "TransformerBlockClassical"]
