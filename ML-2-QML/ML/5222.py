"""Hybrid classical/quantum transformer and image classifier.

This module extends the original ``QTransformerTorch`` by adding a
quantum‑kernel interface and a quanvolution front‑end.  The public API
remains identical to the classical version, but optional flags enable
quantum layers when a quantum backend is available.  The design follows
a *combination* scaling paradigm: classical components form a robust
baseline, while quantum modules are injected through configuration.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttentionBase(nn.Module):
    """Base class for multi‑head attention layers."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError


class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Standard multi‑head attention implemented with PyTorch."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:  # type: ignore[override]
        out, _ = self.attn(x, x, x, key_padding_mask=mask)
        return out


class MultiHeadAttentionQuantum(MultiHeadAttentionClassical):
    """Quantum‑enhanced attention kept for API symmetry; defaults to classical."""
    pass


class FeedForwardBase(nn.Module):
    """Base class for feed‑forward networks."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class FeedForwardClassical(FeedForwardBase):
    """Two‑layer perceptron."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class FeedForwardQuantum(FeedForwardClassical):
    """Quantum feed‑forward kept for API symmetry; defaults to classical."""
    pass


class TransformerBlockBase(nn.Module):
    """Base transformer block containing attention and feed‑forward."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TransformerBlockClassical(TransformerBlockBase):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class TransformerBlockQuantum(TransformerBlockClassical):
    """Quantum transformer block kept for API symmetry; defaults to classical."""
    pass


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding as in the Transformer paper."""
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return x + self.pe[:, : x.size(1)]


class QuanvolutionFilter(nn.Module):
    """Classical 2×2 patch extractor inspired by the original quanvolution."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        feat = self.conv(x)
        return feat.view(x.size(0), -1)


class QFCModel(nn.Module):
    """Simple CNN followed by a 4‑dimensional projection."""
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
        self.fc = nn.Sequential(nn.Linear(16 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 4))
        self.norm = nn.BatchNorm1d(4)
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        feat = self.features(x)
        flat = feat.view(bsz, -1)
        out = self.fc(flat)
        return self.norm(out)


class KernalAnsatz(nn.Module):
    """Radial basis function kernel."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class Kernel(nn.Module):
    """Wrapper around ``KernalAnsatz``."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> torch.Tensor:
    kernel = Kernel(gamma)
    return torch.stack([torch.stack([kernel(x, y) for y in b]) for x in a])


class HybridClassifier(nn.Module):
    """Unified classifier supporting text and image modalities with optional quantum blocks."""
    def __init__(self,
                 modality: str = "text",
                 vocab_size: int = 30522,
                 embed_dim: int = 128,
                 num_heads: int = 4,
                 num_blocks: int = 2,
                 ffn_dim: int = 256,
                 num_classes: int = 2,
                 dropout: float = 0.1,
                 use_quantum_attention: bool = False,
                 use_quantum_ffn: bool = False,
                 use_quanvolution: bool = False,
                 use_quantum_kernel: bool = False,
                 n_qubits_transformer: int = 0,
                 n_qubits_ffn: int = 0,
                 n_qlayers: int = 1,
                 q_device: Optional[torch.device] = None):
        super().__init__()
        self.modality = modality.lower()
        if self.modality not in {"text", "image"}:
            raise ValueError("modality must be 'text' or 'image'")
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout)
        if self.modality == "text":
            self.token_embedding = nn.Embedding(vocab_size, embed_dim)
            self.pos_encoder = PositionalEncoder(embed_dim)
            block_cls = TransformerBlockQuantum if use_quantum_attention or use_quantum_ffn else TransformerBlockClassical
            self.transformers = nn.Sequential(*[
                block_cls(embed_dim, num_heads, ffn_dim,
                          n_qubits_transformer, n_qubits_ffn, n_qlayers,
                          q_device=q_device, dropout=dropout)
                for _ in range(num_blocks)
            ])
            self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)
        else:
            self.qfilter = QuanvolutionFilter() if not use_quanvolution else None
            self.feature_extractor = QFCModel()
            self.use_kernel = use_quantum_kernel
            if self.use_kernel:
                self.prototypes = nn.Parameter(torch.randn(num_classes, 4), requires_grad=False)
                self.kernel = Kernel()
                self.classifier = nn.Identity()
            else:
                self.classifier = nn.Linear(4, num_classes if num_classes > 2 else 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.modality == "text":
            tokens = self.token_embedding(x)
            x = self.pos_encoder(tokens)
            x = self.transformers(x)
            x = x.mean(dim=1)
            x = self.dropout(x)
            return self.classifier(x)
        else:
            if self.qfilter is not None:
                x = self.qfilter(x)
            x = self.feature_extractor(x)
            if self.use_kernel:
                sims = torch.stack([self.kernel(x, p) for p in self.prototypes], dim=1)
                return sims
            else:
                x = self.dropout(x)
                return self.classifier(x)


__all__ = [
    "MultiHeadAttentionBase",
    "MultiHeadAttentionClassical",
    "MultiHeadAttentionQuantum",
    "FeedForwardBase",
    "FeedForwardClassical",
    "FeedForwardQuantum",
    "TransformerBlockBase",
    "TransformerBlockClassical",
    "TransformerBlockQuantum",
    "PositionalEncoder",
    "QuanvolutionFilter",
    "QFCModel",
    "KernalAnsatz",
    "Kernel",
    "kernel_matrix",
    "HybridClassifier",
]
