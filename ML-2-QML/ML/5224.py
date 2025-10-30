"""
Hybrid quantum‑classical classifier – classical implementation.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------
# Basic feed‑forward classifier factory
# ----------------------------------------------------------------------
def build_classifier_circuit(num_features: int, depth: int = 2,
                             dropout: float = 0.0) -> tuple[nn.Module, list[int], list[int], list[int]]:
    """
    Construct a plain feed‑forward network that mirrors the interface of the
    quantum circuit builder.  Returns (network, encoding, weight_sizes, observables).
    """
    layers: list[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: list[int] = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.extend([linear, nn.ReLU(), nn.Dropout(dropout)])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features

    head = nn.Linear(in_dim, 1)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numet())
    network = nn.Sequential(*layers)
    observables = [0]  # dummy placeholder

    return network, encoding, weight_sizes, observables

# ----------------------------------------------------------------------
# Simple transformer utilities
# ----------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    """Classic multi‑head attention based on torch.nn.MultiheadAttention."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_out


class FeedForward(nn.Module):
    """Two‑layer fully connected network."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Standard transformer block."""
    def __init__(self, embed_dim: int, num_heads: int,
                 ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn = FeedForward(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        attn_out = self.attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) *
                             (-np.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor):
        return x + self.pe[:, :x.size(1)]


# ----------------------------------------------------------------------
# Classical RBF kernel utilities
# ----------------------------------------------------------------------
class KernalAnsatz(nn.Module):
    """Placeholder kernel that mimics the quantum interface."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class Kernel(nn.Module):
    """Wraps :class:`KernalAnsatz` for compatibility."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor],
                  gamma: float = 1.0) -> np.ndarray:
    """Compute Gram matrix between two sequences of tensors."""
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])


# ----------------------------------------------------------------------
# Hybrid model definition
# ----------------------------------------------------------------------
class HybridQuantumClassifier(nn.Module):
    """
    Classical implementation that can optionally use transformer layers and
    a classical RBF kernel.  The API mirrors the quantum module so that
    switching implementations is trivial.
    """
    def __init__(self,
                 num_features: int,
                 depth: int = 2,
                 use_transformer: bool = False,
                 transformer_params: Optional[dict] = None,
                 use_kernel: bool = False,
                 gamma: float = 1.0,
                 dropout: float = 0.0):
        super().__init__()
        self.num_features = num_features
        self.depth = depth
        self.use_transformer = use_transformer
        self.use_kernel = use_kernel

        if use_transformer:
            params = transformer_params or {}
            embed_dim = params.get("embed_dim", num_features)
            num_heads = params.get("num_heads", 4)
            num_blocks = params.get("num_blocks", 2)
            ffn_dim = params.get("ffn_dim", 4 * embed_dim)
            self.transformer = nn.Sequential(
                PositionalEncoder(embed_dim),
                *[TransformerBlock(embed_dim, num_heads, ffn_dim,
                                   dropout=dropout) for _ in range(num_blocks)],
                nn.Dropout(dropout)
            )
            self.classifier = nn.Linear(embed_dim, 1)
        else:
            self.network, _, _, _ = build_classifier_circuit(num_features,
                                                             depth,
                                                             dropout=dropout)
            self.classifier = nn.Identity()

        if use_kernel:
            self.kernel = Kernel(gamma)

    def forward(self, x: torch.Tensor):
        """Forward pass through the chosen backbone."""
        if self.use_transformer:
            x = self.transformer(x)
            x = x.mean(dim=1)  # global pooling
        else:
            x = self.network(x)
        return self.classifier(x)

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Expose kernel computation if enabled."""
        if not self.use_kernel:
            raise RuntimeError("Kernel functionality not enabled.")
        return kernel_matrix(a, b, gamma=self.kernel.gamma)


__all__ = [
    "build_classifier_circuit",
    "HybridQuantumClassifier",
    "KernalAnsatz",
    "Kernel",
    "kernel_matrix",
]
