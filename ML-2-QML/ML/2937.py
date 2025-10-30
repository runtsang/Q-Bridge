import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from typing import Sequence

class RBFKernel(nn.Module):
    """Classical RBF kernel with optional trainable gamma."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class KernelMatrix(nn.Module):
    """Wraps the RBF kernel to produce a Gram matrix."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.kernel = RBFKernel(gamma)

    def forward(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return np.array([[self.kernel(x, y).item() for y in b] for x in a])

class TransformerBlock(nn.Module):
    """Standard transformer block."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class UnifiedQuantumKernelTransformer(nn.Module):
    """Hybrid model that uses a classical RBF kernel and a transformer backbone."""
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 kernel_gamma: float = 1.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_enc = PositionalEncoder(embed_dim)
        self.kernel = KernelMatrix(kernel_gamma)
        self.transformer = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, ffn_dim) for _ in range(num_blocks)]
        )
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.embedding(x)
        tokens = self.pos_enc(tokens)
        out = self.transformer(tokens)
        out = out.mean(dim=1)
        return self.classifier(out)
