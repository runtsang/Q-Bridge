"""Fully classical hybrid vision classifier.

This module implements HybridVisionClassifier by combining a CNN backbone,
a classical quanvolution filter, a transformer block, and a sampler network.
It demonstrates how classical sub‑components can be fused into a single
differentiable PyTorch model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# Classical quanvolution filter
class QuanvolutionFilter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)


# Classical transformer components
class MultiHeadAttentionClassical(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_output, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_output


class FeedForwardClassical(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlockClassical(nn.Module):
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


# Classical sampler
class SamplerQNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(inputs), dim=-1)


# Main hybrid classifier
class HybridVisionClassifier(nn.Module):
    """
    Fully classical vision classifier that integrates:
    - Convolutional backbone
    - Classical quanvolution filter on a grayscale projection
    - Transformer block over flattened feature patches
    - Sampler network
    - Final linear classifier
    """
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        # Backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Quanvolution
        self.quanv = QuanvolutionFilter()
        # Transformer
        self.transformer = TransformerBlockClassical(embed_dim=16, num_heads=4, ffn_dim=32)
        # Sampler
        self.sampler = SamplerQNN()
        # Feature projector to 2D for sampler
        self.feature_proj = nn.Linear(16 * 14 * 14 + 4 * 14 * 14, 2)
        # Classifier
        self.classifier = nn.Linear(2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Backbone
        x_backbone = self.backbone(x)                     # (B, 16, H/2, W/2)
        # Flatten for transformer
        B, C, H, W = x_backbone.shape
        patches = x_backbone.view(B, C, -1).transpose(1, 2)  # (B, H*W, C)
        # Transformer
        x_trans = self.transformer(patches)                # (B, H*W, C)
        x_trans = x_trans.mean(dim=1)                      # (B, C)
        # Quanvolution on grayscale projection
        gray = x.mean(dim=1, keepdim=True)                 # (B, 1, H, W)
        q_feat = self.quanv(gray)                          # (B, 4*H/2*W/2)
        # Combine features
        combined = torch.cat((x_trans, q_feat), dim=1)      # (B, 16*14*14 + 4*14*14)
        proj = self.feature_proj(combined)                  # (B, 2)
        sampled = self.sampler(proj)                        # (B, 2)
        logits = self.classifier(sampled)                   # (B, num_classes)
        return logits


__all__ = ["HybridVisionClassifier"]
