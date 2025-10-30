"""Hybrid regression model with classical conv, LSTM and Transformer blocks.

The module mirrors the quantum counterpart while providing a fast classical
implementation.  Each sub‑module can be toggled on/off via constructor flags,
allowing a clean comparison between purely classical and hybrid quantum
regression pipelines.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# --------------------------------------------------------------------------- #
# Data generation and dataset
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate data of the form cos(theta)|0..0⟩ + eᶦφ sin(theta)|1..1⟩.

    Parameters
    ----------
    num_features : int
        Number of features per sample.
    samples : int
        Number of samples to generate.

    Returns
    -------
    x : np.ndarray (samples, num_features)
        Input features.
    y : np.ndarray (samples,)
        Regression target.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset wrapping the superposition data."""

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


# --------------------------------------------------------------------------- #
# Classical sub‑modules
# --------------------------------------------------------------------------- #
class ConvFilter(nn.Module):
    """2‑D convolutional filter acting on reshaped feature vectors."""

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Return a single scalar per sample."""
        tensor = data.view(-1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=[1, 2, 3])


class LSTMEncoder(nn.Module):
    """LSTM encoder that can be used to model sequential dependencies."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return out[:, -1, :]  # last hidden state


class MultiHeadAttention(nn.Module):
    """Standard multi‑head attention."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_linear = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)

        q = q.view(q.size(0), -1, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(k.size(0), -1, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(v.size(0), -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(x.size(0), -1, self.embed_dim)
        return self.out_linear(out)


class FeedForward(nn.Module):
    """Two‑layer feed‑forward network."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float32) *
                             (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerBlock(nn.Module):
    """Transformer block composed of attention and feed‑forward."""

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn = FeedForward(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class TransformerEncoder(nn.Module):
    """Stacked transformer blocks."""

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 num_layers: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, ffn_dim, dropout)
              for _ in range(num_layers)]
        )
        self.pos_encoder = PositionalEncoder(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos_encoder(x)
        return self.blocks(x)


# --------------------------------------------------------------------------- #
# Hybrid regression model
# --------------------------------------------------------------------------- #
class HybridRegressionModel(nn.Module):
    """Hybrid regression model that chains Conv → LSTM → Transformer → Linear.

    Parameters
    ----------
    num_features : int
        Dimensionality of the raw input.
    conv_kernel : int
        Kernel size for the 2‑D convolution filter.
    lstm_hidden : int
        Hidden size of the LSTM encoder.
    transformer_dim : int
        Embedding dimensionality for the transformer.
    transformer_heads : int
        Number of attention heads.
    transformer_ffn : int
        Feed‑forward hidden size.
    transformer_layers : int
        Number of transformer blocks.
    dropout : float
        Dropout probability.
    use_lstm : bool
        Enable the LSTM encoder.
    use_transformer : bool
        Enable the transformer encoder.
    """

    def __init__(self,
                 num_features: int,
                 conv_kernel: int = 2,
                 lstm_hidden: int = 64,
                 transformer_dim: int = 64,
                 transformer_heads: int = 4,
                 transformer_ffn: int = 128,
                 transformer_layers: int = 2,
                 dropout: float = 0.1,
                 use_lstm: bool = True,
                 use_transformer: bool = True):
        super().__init__()
        self.conv = ConvFilter(kernel_size=conv_kernel)
        self.use_lstm = use_lstm
        self.use_transformer = use_transformer

        # The linear projection after convolution
        self.proj = nn.Linear(num_features, transformer_dim)

        if use_lstm:
            self.lstm = LSTMEncoder(num_features, lstm_hidden)
            lstm_out_dim = lstm_hidden
        else:
            lstm_out_dim = num_features

        if use_transformer:
            # Map LSTM output to transformer dimension
            self.lstm_to_transformer = nn.Linear(lstm_out_dim, transformer_dim)
            self.transformer = TransformerEncoder(
                embed_dim=transformer_dim,
                num_heads=transformer_heads,
                ffn_dim=transformer_ffn,
                num_layers=transformer_layers,
                dropout=dropout,
            )
            final_dim = transformer_dim
        else:
            final_dim = lstm_out_dim

        self.head = nn.Linear(final_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convolution
        conv_out = self.conv(x)  # (batch,)
        conv_out = conv_out.unsqueeze(1)  # (batch, 1)

        # Project to embedding space
        embedded = self.proj(conv_out)  # (batch, transformer_dim)

        if self.use_lstm:
            # LSTM expects sequence dimension
            lstm_in = embedded.unsqueeze(1)  # (batch, 1, dim)
            lstm_out = self.lstm(lstm_in)  # (batch, lstm_hidden)
            proj = self.lstm_to_transformer(lstm_out)
        else:
            proj = embedded.squeeze(1)

        if self.use_transformer:
            # Transformer expects sequence length dimension
            transformer_in = proj.unsqueeze(1)  # (batch, 1, transformer_dim)
            transformer_out = self.transformer(transformer_in)  # (batch, 1, transformer_dim)
            feat = transformer_out.squeeze(1)
        else:
            feat = proj

        return self.head(feat).squeeze(-1)


__all__ = [
    "generate_superposition_data",
    "RegressionDataset",
    "HybridRegressionModel",
]
