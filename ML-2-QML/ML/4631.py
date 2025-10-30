"""Hybrid Self‑Attention model – classical implementation.

This module merges a learnable multi‑head self‑attention block with an
RBF kernel feature map and a linear regression head.  It is a drop‑in
replacement for the original `SelfAttention.py` and can be trained
entirely on a CPU or GPU using PyTorch.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import softmax

class SelfAttentionHybrid(nn.Module):
    """Classical hybrid self‑attention + kernel + regression model."""
    def __init__(self, embed_dim: int = 4, n_heads: int = 1, kernel_gamma: float = 1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.kernel_gamma = kernel_gamma
        self.head = nn.Linear(embed_dim, 1)

    def _rbf_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x.unsqueeze(1) - y.unsqueeze(0)
        dist2 = torch.sum(diff * diff, dim=-1)
        return torch.exp(-self.kernel_gamma * dist2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (batch, seq_len, embed_dim)
        attn_out, _ = self.attn(inputs, inputs, inputs)
        features = attn_out.mean(dim=1)  # (batch, embed_dim)
        kernel_mat = self._rbf_kernel(features, features)  # (batch, batch)
        kernel_scores = kernel_mat.diagonal().unsqueeze(-1)  # (batch, 1)
        return self.head(kernel_scores).squeeze(-1)

# ----------------------------------------------------------------------
# Auxiliary data utilities – classical version
# ----------------------------------------------------------------------
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate a toy regression dataset (sin + cos) for classical training."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(torch.utils.data.Dataset):
    """Dataset wrapper around the toy regression data."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

__all__ = ["SelfAttentionHybrid", "RegressionDataset", "generate_superposition_data"]
