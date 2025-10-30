"""Hybrid QCNN model combining convolution, transformer, and fraud‑style scaling."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SimpleSelfAttention(nn.Module):
    """Classical multi‑head self‑attention used in the hybrid transformer block."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.size()
        q = self.q_linear(x).view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(x).view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(x).view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.d_k**0.5
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, seq, self.embed_dim)
        return self.out_linear(out)


class SimpleFeedForward(nn.Module):
    """Position‑wise feed‑forward network."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class SimpleTransformerBlock(nn.Module):
    """Transformer block that can be stacked inside the QCNN."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = SimpleSelfAttention(embed_dim, num_heads, dropout)
        self.ffn = SimpleFeedForward(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class FraudScaleLayer(nn.Module):
    """Linear layer followed by Tanh, scaling and shifting – inspired by the photonic fraud model."""
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.Tanh()
        self.register_buffer("scale", torch.tensor(1.5))
        self.register_buffer("shift", torch.tensor(0.2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.activation(self.linear(x))
        return y * self.scale + self.shift


class QCNNGen130Model(nn.Module):
    """
    Hybrid QCNN architecture that combines:
        * Classical convolution‑style fully‑connected layers (QCNN‑style)
        * A lightweight transformer block for global context
        * A fraud‑style scaling layer for calibrated outputs
    """
    def __init__(self) -> None:
        super().__init__()
        # QCNN‑style feature extraction
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())

        # Transformer block operating on the 4‑dimensional token sequence
        self.transformer = SimpleTransformerBlock(embed_dim=4, num_heads=2, ffn_dim=8)

        # Fraud‑style scaling
        self.fraud_scale = FraudScaleLayer(4, 4)

        # Classification head
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)

        # reshape for transformer: batch × seq_len × embed_dim
        x = x.unsqueeze(1)  # seq_len = 1
        x = self.transformer(x)
        x = x.squeeze(1)

        x = self.fraud_scale(x)
        return torch.sigmoid(self.head(x))


def QCNN() -> QCNNGen130Model:
    """Factory returning a fully‑configured QCNNGen130 model."""
    return QCNNGen130Model()


__all__ = ["QCNNGen130Model", "QCNN"]
