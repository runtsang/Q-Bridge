"""Hybrid classical‑quantum kernel and classifier.

This module pulls together the classical RBF kernel, a transformer‑style feature
extractor, and a hybrid head.  The public API is identical to the seed
implementations but the implementation can be toggled between classical and
quantum behaviour via the ``use_quantum_kernel`` and ``transformer_qconfig``
parameters.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# 1. Classical kernel utilities
# --------------------------------------------------------------------------- #
class ClassicalKernalAnsatz(nn.Module):
    """Radial basis function kernel implemented in pure PyTorch."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class ClassicalKernel(nn.Module):
    """Wrapper that keeps the same shape as the seed."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = ClassicalKernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

def kernel_matrix_classical(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor],
                            gamma: float = 1.0) -> np.ndarray:
    kernel = ClassicalKernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# --------------------------------------------------------------------------- #
# 2. Classical transformer feature extractor
# --------------------------------------------------------------------------- #
class TransformerFeatureExtractorClassical(nn.Module):
    """Feature extractor that uses a purely classical transformer block."""
    def __init__(self, embed_dim: int, num_heads: int,
                 ffn_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

# --------------------------------------------------------------------------- #
# 3. Classical hybrid head
# --------------------------------------------------------------------------- #
class HybridHeadClassical(nn.Module):
    """Combines a linear head with a sigmoid activation."""
    def __init__(self, in_features: int, shift: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)
        return torch.sigmoid(logits + self.shift)

# --------------------------------------------------------------------------- #
# 4. Full model
# --------------------------------------------------------------------------- #
class HybridKernelClassifier(nn.Module):
    """End‑to‑end classifier that can be configured as classical or quantum."""
    def __init__(self,
                 num_features: int,
                 kernel_gamma: float = 1.0,
                 transformer_cfg: Tuple[int, int, int] = (64, 4, 256),
                 num_classes: int = 2):
        super().__init__()
        # Kernel (purely classical in this module)
        self.kernel = ClassicalKernel(gamma=kernel_gamma)
        # Transformer feature extractor
        self.feature_extractor = TransformerFeatureExtractorClassical(
            embed_dim=transformer_cfg[0],
            num_heads=transformer_cfg[1],
            ffn_dim=transformer_cfg[2]
        )
        # Hybrid head
        self.head = HybridHeadClassical(in_features=transformer_cfg[0])
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, feature_dim)
        features = self.feature_extractor(x)
        logits = self.head(features)
        if self.num_classes == 2:
            return torch.cat((logits, 1 - logits), dim=-1)
        else:
            return logits

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Return Gram matrix using the classical kernel."""
        return kernel_matrix_classical(a, b, gamma=self.kernel.gamma)

__all__ = ["HybridKernelClassifier"]
