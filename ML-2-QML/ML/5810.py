"""Hybrid classical quanvolution and self‑attention model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Classical quanvolution filter (2×2 patches → 4 channels)
class ClassicalQuanvolutionFilter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)                     # shape [bsz, 4, 14, 14]
        return features.view(x.size(0), -1)          # shape [bsz, 4*14*14]

# Classical self‑attention helper
class ClassicalSelfAttention:
    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        query = torch.as_tensor(
            inputs @ rotation_params.reshape(self.embed_dim, -1),
            dtype=torch.float32,
        )
        key = torch.as_tensor(
            inputs @ entangle_params.reshape(self.embed_dim, -1),
            dtype=torch.float32,
        )
        value = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()

# Hybrid model
class HybridQuanvolutionAttention(nn.Module):
    """Classical hybrid model combining quanvolution filter and self‑attention."""
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.qfilter = ClassicalQuanvolutionFilter()
        self.attention = ClassicalSelfAttention(embed_dim=4)
        self.linear = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract quanvolution features
        features = self.qfilter(x)                    # [bsz, 4*14*14]
        # Convert to numpy for the attention routine
        features_np = features.detach().cpu().numpy()
        # Random parameters (placeholder) – in practice learnable or data‑driven
        rotation_params = np.random.randn(4 * 3)
        entangle_params = np.random.randn(3)
        # Apply self‑attention
        attended = self.attention.run(rotation_params, entangle_params, features_np)
        attended_tensor = torch.as_tensor(attended, device=x.device, dtype=torch.float32)
        logits = self.linear(attended_tensor)
        return F.log_softmax(logits, dim=-1)

__all__ = ["HybridQuanvolutionAttention"]
