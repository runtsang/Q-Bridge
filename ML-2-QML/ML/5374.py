"""Classical hybrid binary classifier that combines CNN, self‑attention, and an Estimator‑style head.

The architecture mirrors the quantum counterpart but remains fully classical.  The
self‑attention block operates on a small slice of the flattened feature map and
produces an additional feature that is concatenated with the rest of the
representation before the final Estimator head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ClassicalSelfAttention(nn.Module):
    """Scaled‑dot‑product self‑attention implemented with pure PyTorch."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq, embed)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.embed_dim)
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, v)

class EstimatorHead(nn.Module):
    """Feed‑forward head inspired by the EstimatorQNN example."""
    def __init__(self, in_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class HybridBinaryClassifier(nn.Module):
    """CNN → self‑attention → fully‑connected → Estimator head."""
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.5),
        )
        # size after conv layers for a 32x32 input: 15 * 7 * 7 = 735
        self.attn = ClassicalSelfAttention(embed_dim=735)
        self.fc = nn.Sequential(
            nn.Linear(735 + 1, 120),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
        )
        self.head = EstimatorHead(in_features=84)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)  # (batch, 735)
        # apply self‑attention on a single‑token sequence
        seq = x.unsqueeze(1)  # (batch, 1, 735)
        attn_out = self.attn(seq).squeeze(1)  # (batch, 735)
        # aggregate attention as a scalar feature
        attn_scalar = attn_out.mean(dim=-1, keepdim=True)  # (batch, 1)
        x = torch.cat([x, attn_scalar], dim=-1)  # (batch, 736)
        x = self.fc(x)
        logits = self.head(x)
        probs = torch.sigmoid(logits)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["HybridBinaryClassifier"]
