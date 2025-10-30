from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class QuanvolutionFilter(nn.Module):
    """Classical 2×2 convolutional filter producing 4 channel outputs."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding for transformer tokens."""
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

class TransformerBlock(nn.Module):
    """Standard transformer block with multi‑head attention and feed‑forward."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class HybridFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float):
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        outputs, = ctx.saved_tensors
        grad_inputs = grad_output * outputs * (1 - outputs)
        return grad_inputs, None

class Hybrid(nn.Module):
    """Linear head followed by a sigmoid with a trainable shift."""
    def __init__(self, in_features: int, shift: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = inputs.view(inputs.size(0), -1)
        return HybridFunction.apply(self.linear(logits), self.shift)

class EstimatorQNN(nn.Module):
    """Small feed‑forward regressor."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)

class QuanvolutionNet(nn.Module):
    """
    Hybrid architecture that combines a classical quanvolution filter, a transformer encoder,
    and a flexible head that can perform classification, regression or a quantum‑style hybrid output.
    """
    def __init__(
        self,
        num_classes: int = 10,
        transformer_blocks: int = 2,
        embed_dim: int = 64,
        num_heads: int = 4,
        ffn_dim: int = 128,
        shift: float = 0.0,
        use_hybrid_head: bool = False,
        use_regressor: bool = False
    ):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        self.qfilter = QuanvolutionFilter()
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.transformers = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, ffn_dim) for _ in range(transformer_blocks)]
        )
        self.dropout = nn.Dropout(0.1)
        self.use_hybrid_head = use_hybrid_head
        self.use_regressor = use_regressor

        # projection from 4‑channel patches to embedding space
        self.proj = nn.Linear(4, embed_dim)

        if use_regressor:
            self.regressor_proj = nn.Linear(embed_dim, 2)
            self.head = EstimatorQNN()
        elif use_hybrid_head:
            self.head = Hybrid(embed_dim, shift=shift)
        else:
            self.linear = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # classical conv to reduce resolution
        x = self.conv(x)                 # (B,4,14,14)
        # quanvolution filter (classical)
        patches = self.qfilter(x)        # (B,4,14,14)
        # reshape to (B, seq_len, 4)
        patches = patches.permute(0, 2, 3, 1).reshape(x.size(0), -1, 4)
        # embed patches
        x = self.proj(patches)           # (B, seq_len, embed_dim)
        # add positional encoding
        x = self.pos_encoder(x)
        # transformer encoder
        x = self.transformers(x)
        # aggregate tokens
        x = self.dropout(x.mean(dim=1))
        if self.use_regressor:
            x = self.regressor_proj(x)
            return self.head(x)
        elif self.use_hybrid_head:
            return self.head(x)
        else:
            logits = self.linear(x)
            return F.log_softmax(logits, dim=-1)

__all__ = [
    "QuanvolutionNet",
    "HybridFunction",
    "Hybrid",
    "EstimatorQNN",
    "QuanvolutionFilter",
    "TransformerBlock",
    "PositionalEncoder"
]
