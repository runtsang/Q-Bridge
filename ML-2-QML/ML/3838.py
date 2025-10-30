"""
HybridFusionNet: A modular PyTorch implementation that fuses CNN, transformer, and quantum‑enabled heads.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
#  Classical utilities
# --------------------------------------------------------------------------- #

class HybridFunction(nn.Module):
    """
    Differentiable wrapper that maps a scalar through a sigmoid head.
    Replaces the original QuantumExpectation head in the seed.
    """
    def __init__(self, shift: float = 0.0) -> None:
        super().__init__()
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x + self.shift)


class HybridHead(nn.Module):
    """
    Dense head that feeds into a HybridFunction.
    Supports optional scaling of the feature vector before the sigmoid.
    """
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, 1)
        self.act = HybridFunction(shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.fc(x))


# --------------------------------------------------------------------------- #
#  CNN backbone
# --------------------------------------------------------------------------- #

class ConvBackbone(nn.Module):
    """
    Feature extractor mirroring the convolutional layers of the seed.
    Keeps the flattened feature dimension (55815) for compatibility.
    """
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        return self.drop2(x)


# --------------------------------------------------------------------------- #
#  Transformer‑style attention block
# --------------------------------------------------------------------------- #

class MultiHeadAttentionMixin:
    """
    Shared logic for attention heads (classical or quantum).
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, dim = x.size()
        return x.view(batch, seq, self.num_heads, dim // self.num_heads).transpose(1, 2)

    def _merge_heads(self, heads: torch.Tensor) -> torch.Tensor:
        batch, heads_, seq, dim = heads.size()
        return heads.transpose(1, 2).contiguous().view(batch, seq, heads_ * dim)


class MultiHeadAttentionClassical(MultiHeadAttentionMixin):
    """Standard multi‑head attention implemented purely classically."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__(embed_dim, num_heads, dropout)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (self._split_heads(t) for t in qkv)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.embed_dim // self.num_heads)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        return self.out(self._merge_heads(out))


class MultiHeadAttentionQuantum(MultiHeadAttentionMixin):
    """
    Quantum‑enabled attention that runs each head through a small parametric circuit.
    Uses a lightweight Pennylane‑like interface for illustration.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__(embed_dim, num_heads, dropout)
        self.n_wires = num_heads
        self.q_params = nn.Parameter(torch.randn(num_heads, 1, 1))  # placeholder

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq, _ = x.size()
        # simple quantum‑style embedding: rotate each head
        q = x[:, :, :self.n_wires]
        # simulate a rotation with learnable params
        q = torch.cos(self.q_params) * q + torch.sin(self.q_params) * (x[:, :, self.n_wires:])
        return q.reshape(batch, seq, -1)


class TransformerBlock(nn.Module):
    """
    A single transformer block that can swap its attention or feed‑forward part
    between classical and quantum variants.
    """
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 use_quantum_attn: bool = False,
                 use_quantum_ffn: bool = False):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.attn = (MultiHeadAttentionQuantum if use_quantum_attn
                     else MultiHeadAttentionClassical)(embed_dim, num_heads)
        self.ffn = (self._quantum_ffn if use_quantum_ffn
                     else self._classical_ffn)(embed_dim, ffn_dim)

    def _classical_ffn(self, embed_dim: int, ffn_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embed_dim)
        )

    def _quantum_ffn(self, embed_dim: int, ffn_dim: int) -> nn.Module:
        # simplified quantum feed‑forward: linear + parameterized rotation
        return nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.Dropout(0.1),
            nn.Linear(ffn_dim, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class PositionalEncoder(nn.Module):
    """
    Sinusoidal positional encoding that matches the seed implementation.
    """
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# --------------------------------------------------------------------------- #
#  Final classifier
# --------------------------------------------------------------------------- #

class HybridFusionNet(nn.Module):
    """
    Fusion model that stitches together:
    1) ConvBackbone – image feature extractor
    2) TransformerBlock layer that can use quantum heads or feed‑forward
    3) HybridHead – classical dense layer followed by sigmoid
    """
    def __init__(self,
                 embed_dim: int = 120,
                 num_heads: int = 8,
                 ffn_dim: int = 256,
                 num_blocks: int = 2,
                 use_quantum_attn: bool = False,
                 use_quantum_ffn: bool = False):
        super().__init__()
        self.backbone = ConvBackbone()
        self.transformer = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, ffn_dim,
                               use_quantum_attn, use_quantum_ffn)
              for _ in range(num_blocks)])
        self.head = HybridHead(in_features=embed_dim, shift=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. CNN feature extraction
        x = self.backbone(x)            # (B, 3, 0..)
        # 2. Flatten and project to embed_dim
        #   The seed used 55815 -> 120 -> 84 -> 1; we keep the same linear chain
        #   but replace the fully‑connected layers with a *single* linear layer
        #   that matches one‑hot encodings of image‑based feature vectors.
        flattened = torch.flatten(x, 1)
        # <--- insert conditional forward path for 1‑D convolution
        # 3. **transformer** (??)   [custom]
        #   (treat image features as token sequence)
        #   The logic is (....).
