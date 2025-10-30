from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

class HybridSelfAttention(nn.Module):
    """Classical hybrid self‑attention + convolutional network."""
    def __init__(self, embed_dim: int = 4, in_channels: int = 1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.attention = self._build_attention(embed_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
        )
        self.norm = nn.BatchNorm1d(embed_dim)

    @staticmethod
    def _build_attention(embed_dim: int) -> nn.Module:
        class SelfAttentionLayer(nn.Module):
            def __init__(self, dim: int) -> None:
                super().__init__()
                self.query = nn.Linear(dim, dim, bias=False)
                self.key = nn.Linear(dim, dim, bias=False)
                self.value = nn.Linear(dim, dim, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                q = self.query(x)
                k = self.key(x)
                v = self.value(x)
                scores = torch.softmax(q @ k.transpose(-2, -1) / np.sqrt(q.size(-1)), dim=-1)
                return torch.matmul(scores, v)

        return SelfAttentionLayer(embed_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs : torch.Tensor
            Input image batch of shape (batch, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Normalised self‑attention output of shape (batch, embed_dim).
        """
        feats = self.cnn(inputs)
        flat = feats.view(feats.size(0), -1)
        proj = self.fc(flat)
        attn_out = self.attention(proj)
        return self.norm(attn_out)

def HybridSelfAttention_factory() -> HybridSelfAttention:
    return HybridSelfAttention()

__all__ = ["HybridSelfAttention", "HybridSelfAttention_factory"]
