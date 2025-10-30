"""Hybrid classical-quantum Natural Language Model combining CNN, self-attention, and quantum-inspired layers."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassicalAttention(nn.Module):
    """Self-attention module with learnable query/key/value projections."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query_proj(x)
        k = self.key_proj(x)
        v = self.value_proj(x)
        scores = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.embed_dim), dim=-1)
        return scores @ v

class HybridNATModel(nn.Module):
    """Classical CNN + self‑attention + fully‑connected projection to 4 features."""
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Self‑attention branch on the pooled representation
        self.attention = ClassicalAttention(embed_dim=16)
        self.attn_proj = nn.Linear(16, 16 * 7 * 7)
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        feats = self.features(x)                       # (bsz, 16, 7, 7)
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)       # (bsz, 16)
        attn_out = self.attention(pooled)              # (bsz, 16)
        flat = feats.view(bsz, -1)                     # (bsz, 16*7*7)
        fused = flat + self.attn_proj(attn_out)        # Fuse attention into features
        out = self.fc(fused)
        return self.norm(out)

__all__ = ["HybridNATModel"]
