"""Hybrid classical convolutional transformer module.

Provides a drop‑in replacement for the original Conv filter, but
augments it with a transformer stack that can be configured to use
classical or quantum sub‑components.  The class exposes a unified
``Conv`` function returning an instance ready for training or inference.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np

class HybridConvTransformer(nn.Module):
    """Classical hybrid model: conv filter + transformer + self‑attention."""
    def __init__(
        self,
        mode: str = "classical",
        kernel_size: int = 2,
        threshold: float = 0.0,
        embed_dim: int = 128,
        num_heads: int = 4,
        ffn_dim: int = 256,
        num_blocks: int = 2,
        n_qubits_transformer: int = 0,
        n_qubits_ffn: int = 0,
        n_qlayers: int = 1,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.kernel_size = kernel_size
        self.threshold = threshold
        if mode == "classical":
            self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
            self.transformer = nn.Sequential(
                *[self._build_transformer_block(embed_dim, num_heads, ffn_dim) for _ in range(num_blocks)]
            )
            self.sampler = nn.Sequential(
                nn.Linear(embed_dim, 4),
                nn.Tanh(),
                nn.Linear(4, 2),
            )
        else:
            # Quantum mode is handled in the QML implementation.
            self.conv = None
            self.transformer = None
            self.sampler = None

    def _build_transformer_block(self, embed_dim: int, num_heads: int, ffn_dim: int):
        attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        ff = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embed_dim),
        )
        norm1 = nn.LayerNorm(embed_dim)
        norm2 = nn.LayerNorm(embed_dim)
        dropout = nn.Dropout(0.1)

        def block(x: torch.Tensor) -> torch.Tensor:
            attn_out, _ = attn(x, x, x)
            x = norm1(x + dropout(attn_out))
            ffn_out = ff(x)
            x = norm2(x + dropout(ffn_out))
            return x

        return block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode!= "classical":
            raise RuntimeError("Quantum mode is not supported in the classical implementation")
        # Convolution
        x = self.conv(x)
        x = torch.sigmoid(x - self.threshold)
        # Flatten to sequence
        seq = x.view(x.size(0), -1, self.conv.out_channels)
        # Transformer
        for block in self.transformer:
            seq = block(seq)
        # Pooling
        out = seq.mean(dim=1)
        # Probabilistic sampler
        probs = F.softmax(self.sampler(out), dim=-1)
        # Return probability of the second class
        return probs[:, 1]

def Conv() -> HybridConvTransformer:
    """Return a hybrid classical Conv‑Transformer instance."""
    return HybridConvTransformer(mode="classical")
