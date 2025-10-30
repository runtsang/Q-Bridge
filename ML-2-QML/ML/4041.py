"""UnifiedEstimatorQNN: classical implementation.

Provides three backends:
- ``classical``: shallow feed‑forward regressor.
- ``kernel``: kernel ridge regression with a classical RBF kernel.
- ``transformer``: transformer‑based classifier (classical transformer).

All backends expose a common API: ``forward(x)``, ``predict(x)`` and ``fit(X, y)``.
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# 1. Feed‑forward regressor
# --------------------------------------------------------------------------- #
class EstimatorNN(nn.Module):
    """Shallow fully‑connected regression network."""
    def __init__(self, input_dim: int = 2, hidden_sizes: Optional[List[int]] = None) -> None:
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [8, 4]
        layers = []
        last = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.Tanh())
            last = h
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# --------------------------------------------------------------------------- #
# 2. Classical RBF kernel
# --------------------------------------------------------------------------- #
class ClassicalKernel(nn.Module):
    """Radial‑basis function kernel."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x: (n, d), y: (m, d)
        diff = x.unsqueeze(1) - y.unsqueeze(0)  # (n, m, d)
        sq = (diff ** 2).sum(-1)  # (n, m)
        return torch.exp(-self.gamma * sq)

def kernel_matrix(a: List[torch.Tensor], b: List[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    k = ClassicalKernel(gamma)
    return np.array([[k(x, y).item() for y in b] for x in a])

# --------------------------------------------------------------------------- #
# 3. Kernel ridge regressor
# --------------------------------------------------------------------------- #
class KernelRegressor(nn.Module):
    """Kernel ridge regression using the classical RBF kernel."""
    def __init__(self, gamma: float = 1.0, lambda_reg: float = 1e-3) -> None:
        super().__init__()
        self.gamma = gamma
        self.lambda_reg = lambda_reg
        self.train_X = None
        self.train_y = None
        self.alpha = None

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        self.train_X, self.train_y = X, y
        K = ClassicalKernel(self.gamma)(X, X) + self.lambda_reg * torch.eye(X.size(0))
        self.alpha = torch.linalg.solve(K, y)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.alpha is None:
            raise RuntimeError("Model not fitted")
        K_test = ClassicalKernel(self.gamma)(X, self.train_X)
        return K_test @ self.alpha

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return self.forward(X)

# --------------------------------------------------------------------------- #
# 4. Classical transformer
# --------------------------------------------------------------------------- #
class MultiHeadAttention(nn.Module):
    """Standard multi‑head attention."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_out

class FeedForward(nn.Module):
    """Two‑layer feed‑forward network."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlock(nn.Module):
    """Transformer block with residual connections."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn = FeedForward(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out = self.attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]

class TextClassifier(nn.Module):
    """Transformer‑based text classifier."""
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.transformers = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, ffn_dim, dropout) for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_encoder(tokens)
        x = self.transformers(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)

# --------------------------------------------------------------------------- #
# 5. Unified estimator
# --------------------------------------------------------------------------- #
class UnifiedEstimatorQNN(nn.Module):
    """Hybrid estimator with interchangeable backends.

    Parameters
    ----------
    mode : {'classical', 'kernel', 'transformer'}
        Backend to use.
    kwargs : dict
        Additional arguments forwarded to the chosen backend.
    """
    def __init__(self, mode: str = "classical", **kwargs) -> None:
        super().__init__()
        self.mode = mode
        if mode == "classical":
            self.backend = EstimatorNN(**kwargs)
        elif mode == "kernel":
            self.backend = KernelRegressor(**kwargs)
        elif mode == "transformer":
            self.backend = TextClassifier(**kwargs)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backend(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        if hasattr(self.backend, "fit"):
            self.backend.fit(X, y)
        else:
            raise AttributeError("Backend does not implement fit")

__all__ = ["UnifiedEstimatorQNN", "EstimatorNN", "KernelRegressor", "TextClassifier"]
