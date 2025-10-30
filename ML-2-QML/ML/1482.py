"""Extended classical convolutional filter with attention for MNIST-like data."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    """Additive attention over patch embeddings."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.attn = nn.Linear(embed_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, num_patches, embed_dim)
        Returns:
            context: Tensor of shape (batch, embed_dim)
        """
        scores = self.attn(x)  # (batch, num_patches, 1)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(weights * x, dim=1)
        return context

class QuanvolutionAdvanced(nn.Module):
    """Classical hybrid model: patch embedding via Conv2d, attention, linear classifier."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.patch_embed = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        self.attention = AttentionLayer(embed_dim=4)
        self.classifier = nn.Linear(4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, 1, 28, 28)
        Returns:
            log probabilities: Tensor of shape (batch, num_classes)
        """
        patches = self.patch_embed(x)  # (batch, 4, 14, 14)
        patches = patches.view(patches.size(0), 4, -1).permute(0, 2, 1)  # (batch, 14*14, 4)
        context = self.attention(patches)  # (batch, 4)
        logits = self.classifier(context)  # (batch, num_classes)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionAdvanced"]
