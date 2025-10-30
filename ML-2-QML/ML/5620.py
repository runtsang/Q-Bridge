"""Hybrid classical quanvolution network with optional transformer."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMultiHeadAttention(nn.Module):
    """Classical multi‑head attention using PyTorch's MultiheadAttention."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.attn(x, x, x, key_padding_mask=mask)[0]


class SimpleFeedForward(nn.Module):
    """Two‑layer feed‑forward network."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class SimpleTransformerBlock(nn.Module):
    """Transformer block that can be stacked."""

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.attn = SimpleMultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn = SimpleFeedForward(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class HybridQuanvolutionNet(nn.Module):
    """
    Classical hybrid network that mimics the original quanvolution architecture
    but optionally augments the feature stream with a transformer stack.
    """

    def __init__(self,
                 num_classes: int = 10,
                 use_transformer: bool = False,
                 transformer_layers: int = 2,
                 transformer_heads: int = 4,
                 transformer_ffn_dim: int = 256) -> None:
        super().__init__()
        # Classical filter
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

        # Linear projection of flattened features
        self.fc = nn.Linear(4 * 14 * 14, 128)
        self.bn = nn.BatchNorm1d(128)

        self.use_transformer = use_transformer
        if use_transformer:
            # Stack several transformer blocks
            self.transformer = nn.Sequential(
                *[SimpleTransformerBlock(128, transformer_heads, transformer_ffn_dim)
                  for _ in range(transformer_layers)]
            )
        else:
            self.transformer = None

        # Final classifier
        self.classifier = nn.Linear(128, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature extraction
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        x = self.bn(x)

        # Optional transformer processing
        if self.transformer:
            # Treat the 128‑dim vector as a one‑token sequence
            seq = x.unsqueeze(1)
            seq = self.transformer(seq)
            x = seq.squeeze(1)

        logits = self.classifier(x)
        return F.log_softmax(logits, dim=-1)


__all__ = ["HybridQuanvolutionNet"]
