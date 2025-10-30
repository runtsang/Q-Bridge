"""Hybrid transformer module that keeps the original API but adds a layer‑wise quantum flag and a quantum embedding option."""

from __future__ import annotations

import math
from typing import Callable, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttentionBase(nn.Module):
    """Base class for all attention implementations."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, *args, **kwargs) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Standard multi‑head attention implemented with nn.MultiheadAttention."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self._attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out, _ = self._attn(x, x, x, key_padding_mask=mask)
        return attn_out


class FeedForwardBase(nn.Module):
    """Base class for feed‑forward layers."""
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
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class TransformerBlockClassical(TransformerBlockBase):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
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
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class QuantumEmbedding(nn.Module):
    """A lightweight classical 'quantum' embedding that applies an orthogonal transform."""
    def __init__(self, embed_dim: int, n_wires: int = 8, bias: bool = False) -> None:
        super().__init__()
        self.n_wires = n_wires
        # Linear layer that will be orthogonally initialized
        self.linear = nn.Linear(embed_dim, embed_dim, bias=bias)
        self._init_orthogonal()

    def _init_orthogonal(self) -> None:
        with torch.no_grad():
            ortho = torch.qr(torch.randn(self.linear.weight.size()))[0]
            self.linear.weight.copy_(ortho)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class HybridTransformerBlock(nn.Module):
    """
    Transformer block that can operate either classically or with a quantum replacement.
    The quantum replacement is a placeholder that simply forwards the input; the real
    quantum implementation lives in the QML module.
    """
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 use_quantum: bool = False,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.use_quantum = use_quantum
        if use_quantum:
            self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
            self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)
        else:
            self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
            self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class TextClassifier(nn.Module):
    """
    Original transformer‑based text classifier.
    """
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 n_qubits_transformer: int = 0,
                 n_qubits_ffn: int = 0,
                 n_qlayers: int = 1,
                 q_device: Optional[object] = None) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        self.transformers = nn.Sequential(
            *[TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout) for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        x = self.transformers(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)


class HybridTextClassifier(TextClassifier):
    """
    Extended classifier that can enable a layer‑wise quantum flag and a quantum embedding.
    """
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 quantum_per_block: Optional[List[bool]] = None,
                 use_quantum_embedding: bool = False,
                 n_qubits_transformer: int = 0,
                 n_qubits_ffn: int = 0,
                 n_qlayers: int = 1,
                 q_device: Optional[object] = None) -> None:
        super().__init__(vocab_size, embed_dim, num_heads, num_blocks,
                         ffn_dim, num_classes, dropout,
                         n_qubits_transformer, n_qubits_ffn, n_qlayers, q_device)
        # Override token embedding if requested
        if use_quantum_embedding:
            self.token_embedding = QuantumEmbedding(embed_dim)
        # Override transformer blocks
        if quantum_per_block is None:
            quantum_per_block = [False] * num_blocks
        blocks = []
        for use_q in quantum_per_block:
            blocks.append(HybridTransformerBlock(embed_dim, num_heads, ffn_dim,
                                                 use_quantum=use_q,
                                                 dropout=dropout))
        self.transformers = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        x = self.transformers(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)


def pretrain_autoencoder(encoder: nn.Module,
                         tokenizer: Callable[[str], torch.Tensor],
                         dataset: Iterable[str],
                         epochs: int = 5,
                         lr: float = 1e-3) -> None:
    """
    Lightweight auto‑encoding pre‑training loop that trains the encoder to reconstruct
    the input token sequence.  The encoder is typically a TextClassifier or its hybrid
    variant; only the embedding and positional layers are used as the encoder and the
    classification head is ignored.
    """
    device = next(encoder.parameters()).device
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    encoder.train()
    for epoch in range(epochs):
        for text in dataset:
            tokens = tokenizer(text).to(device)
            optimizer.zero_grad()
            logits = encoder(tokens)
            # The raw logits correspond to the classification output; for auto‑encoding
            # we pretend each token is a target class and compute cross‑entropy against
            # the original token indices.
            loss = criterion(logits, tokens)
            loss.backward()
            optimizer.step()
        print(f"Auto‑encoder epoch {epoch+1}/{epochs} finished.")


__all__ = [
    "MultiHeadAttentionBase",
    "MultiHeadAttentionClassical",
    "FeedForwardBase",
    "FeedForwardClassical",
    "TransformerBlockBase",
    "TransformerBlockClassical",
    "PositionalEncoder",
    "QuantumEmbedding",
    "HybridTransformerBlock",
    "TextClassifier",
    "HybridTextClassifier",
    "pretrain_autoencoder",
]
