"""Hybrid transformer with optional quantum‑inspired modules."""

from __future__ import annotations

import math
import numpy as np
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------
# 1. Classical Self‑Attention helper (from reference 2)
# ------------------------------------------------------------------
class ClassicalSelfAttention:
    """Simple self‑attention implementation used as a drop‑in replacement
    for the quantum SelfAttention circuit.  It accepts rotation and
    entangle parameters and returns a weighted sum of the input
    embeddings.  The parameters are interpreted as linear
    transformations for query/key/value generation."""
    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim
        # Parameters are randomly initialised; in a real experiment they
        # would be optimised jointly with the rest of the network.
        self.rotation_params = np.random.randn(embed_dim * 3)
        self.entangle_params = np.random.randn(embed_dim - 1)

    def run(self, inputs: np.ndarray) -> np.ndarray:
        """Return the attention output for a batch of sequences.

        Parameters
        ----------
        inputs
            Array of shape (batch, seq_len, embed_dim).

        Returns
        -------
        output
            Array of shape (batch, seq_len, embed_dim) after applying
            attention weights.
        """
        batch, seq_len, embed_dim = inputs.shape
        # Flatten batch and seq to matrix multiplication
        flat = inputs.reshape(-1, embed_dim)
        query = flat @ self.rotation_params.reshape(embed_dim, -1)
        key = flat @ self.entangle_params.reshape(embed_dim, -1)
        value = flat
        scores = torch.softmax(torch.matmul(
            query, key.T) / math.sqrt(embed_dim), dim=-1)
        out = torch.matmul(scores, value)
        return out.reshape(batch, seq_len, embed_dim).numpy()


# ------------------------------------------------------------------
# 2. Classical CNN‑FC feature extractor (from reference 3)
# ------------------------------------------------------------------
class QFCModel(nn.Module):
    """Convolutional feature extractor followed by a 4‑dimensional
    projection.  It matches the classical part of the Quantum‑NAT
    example and can be used as a pre‑processor for image inputs."""
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
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 4)
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        features = self.features(x)
        flattened = features.view(bsz, -1)
        out = self.fc(flattened)
        return self.norm(out)


# ------------------------------------------------------------------
# 3. Transformer primitives (modified from reference 1)
# ------------------------------------------------------------------
class MultiHeadAttentionBase(nn.Module):
    """Shared logic for attention layers."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Standard multi‑head attention implemented using PyTorch."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_output, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_output


class FeedForwardBase(nn.Module):
    """Base class for the position‑wise feed‑forward network."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class FeedForwardClassical(FeedForwardBase):
    """Two‑layer perceptron feed‑forward network."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlockBase(nn.Module):
    """Base transformer block containing attention and feed‑forward parts."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class TransformerBlockClassical(TransformerBlockBase):
    """Classical transformer block."""
    def __init__(self, embed_dim: int, num_heads: int,
                 ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# ------------------------------------------------------------------
# 4. Hybrid transformer (classical + optional quantum‑inspired parts)
# ------------------------------------------------------------------
class HybridTransformerClassic(nn.Module):
    """Hybrid transformer that can optionally use a classical
    SelfAttention block or a CNN feature extractor.

    Parameters
    ----------
    vocab_size
        Size of the token vocabulary.
    embed_dim
        Dimensionality of token embeddings.
    num_heads
        Number of attention heads.
    num_blocks
        Number of transformer blocks.
    ffn_dim
        Dimensionality of the inner feed‑forward layer.
    num_classes
        Number of output classes.  If ``num_classes`` <= 2 the model
        outputs a single logit.
    dropout
        Drop‑out probability.
    use_self_attention
        If ``True`` the attention mechanism is replaced with the
        ClassicalSelfAttention helper from reference 2.  The helper
        operates on the raw embeddings and returns a weighted sum
        that is fed through the remaining transformer layers.
    use_feature_extractor
        If ``True`` the input is treated as an image and first passed
        through the QFCModel (reference 3) before tokenisation.
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        use_self_attention: bool = False,
        use_feature_extractor: bool = False,
    ) -> None:
        super().__init__()
        self.use_self_attention = use_self_attention
        self.use_feature_extractor = use_feature_extractor

        if use_feature_extractor:
            self.feature_extractor = QFCModel()
            # Map the 4‑dimensional output of QFCModel to the embedding space
            self.feature_proj = nn.Linear(4, embed_dim)
        else:
            self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        self.pos_embedding = PositionalEncoder(embed_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlockClassical(
                    embed_dim, num_heads, ffn_dim, dropout
                )
                for _ in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout)

        if num_classes > 2:
            self.classifier = nn.Linear(embed_dim, num_classes)
        else:
            self.classifier = nn.Linear(embed_dim, 1)

        if use_self_attention:
            # A single instance of ClassicalSelfAttention is enough; the
            # parameters are random but can be optimised by adding a
            # learning rate schedule if desired.
            self.self_attention = ClassicalSelfAttention(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            * If ``use_feature_extractor`` is ``True``: a tensor of shape
              ``(batch, 1, H, W)`` representing a grayscale image.
            * Otherwise: an integer tensor of shape ``(batch, seq_len)``.
        """
        if self.use_feature_extractor:
            # Image → feature vector (batch, 4)
            feats = self.feature_extractor(x)  # (batch, 4)
            # Project to embedding space
            x = self.feature_proj(feats).unsqueeze(1)  # (batch, 1, embed_dim)
        else:
            x = self.token_embedding(x)  # (batch, seq_len, embed_dim)

        x = self.pos_embedding(x)

        if self.use_self_attention:
            # Apply the classical self‑attention helper before the
            # transformer blocks.  The helper returns a NumPy array so we
            # convert back to a torch tensor.
            attn_out = torch.tensor(
                self.self_attention.run(x.detach().cpu().numpy()),
                dtype=x.dtype,
                device=x.device,
            )
            x = attn_out

        for block in self.transformer_blocks:
            x = block(x)

        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)


__all__ = [
    "HybridTransformerClassic",
    "SelfAttention",
    "QFCModel",
    "MultiHeadAttentionBase",
    "MultiHeadAttentionClassical",
    "FeedForwardBase",
    "FeedForwardClassical",
    "TransformerBlockBase",
    "TransformerBlockClassical",
    "PositionalEncoder",
]
