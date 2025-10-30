import math
from typing import Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_classical_classifier(
    num_features: int,
    depth: int,
    hidden_size: int | None = None,
) -> Tuple[nn.Sequential, List[int], List[int], List[int]]:
    """Construct a purely classical feed‑forward classifier.

    Parameters
    ----------
    num_features: int
        Dimensionality of the input feature vector.
    depth: int
        Number of hidden layers to stack.
    hidden_size: int | None
        Size of each hidden layer.  If ``None`` the hidden size is
        set equal to ``num_features``.

    Returns
    -------
    network: nn.Sequential
        The classifier network.
    encoding: List[int]
        Indices of input features that are directly passed to the
        network (identity mapping).
    weight_sizes: List[int]
        Number of parameters in each linear layer.
    observables: List[int]
        Dummy observables; mirrors the quantum interface but has no
        functional effect in the classical case.
    """
    if hidden_size is None:
        hidden_size = num_features

    layers: List[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: List[int] = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, hidden_size)
        layers.extend([linear, nn.ReLU()])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = hidden_size

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables = [0, 1]  # placeholder for compatibility
    return network, encoding, weight_sizes, observables


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding as in Vaswani et al."""

    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class MultiHeadAttention(nn.Module):
    """Standard multi‑head self‑attention."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)
        return self.dropout(attn_out)


class FeedForward(nn.Module):
    """Two‑layer MLP with ReLU."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Single transformer block (attention + FFN)."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn = FeedForward(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        return self.norm2(x + ffn_out)


class HybridTransformerClassifier(nn.Module):
    """
    A transformer‑based text classifier that can be instantiated with a
    purely classical backbone.  The class mirrors the API of the
    original quantum variant but keeps all operations on the CPU/GPU.

    Parameters
    ----------
    vocab_size: int
        Size of the token vocabulary.
    embed_dim: int
        Dimensionality of token embeddings.
    num_heads: int
        Number of attention heads.
    num_blocks: int
        Number of transformer blocks.
    ffn_dim: int
        Hidden size of the feed‑forward sub‑network.
    num_classes: int
        Number of target classes.
    dropout: float
        Drop‑out probability.
    ffn_depth: int
        Optional depth of the feed‑forward head (default 1).
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
        ffn_depth: int = 1,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, ffn_dim, dropout)
                for _ in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)

        self.ffn_depth = ffn_depth
        if ffn_depth > 1:
            hidden = ffn_dim
            self.head = nn.Sequential(
                *[
                    nn.Linear(embed_dim, hidden),
                    nn.ReLU(),
                ]
                * (ffn_depth - 1)
                + [nn.Linear(hidden, num_classes)],
            )
        else:
            self.head = self.classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            Input token indices of shape (batch, seq_len).

        Returns
        -------
        torch.Tensor
            Logits of shape (batch, num_classes).
        """
        x = self.token_embedding(x)
        x = self.pos_encoder(x)
        for blk in self.blocks:
            x = blk(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.head(x)

    def get_weight_sizes(self) -> List[int]:
        """Return a list of parameter counts for each linear layer."""
        sizes: List[int] = []
        for m in self.modules():
            if isinstance(m, nn.Linear):
                sizes.append(m.weight.numel() + m.bias.numel())
        return sizes


__all__ = [
    "HybridTransformerClassifier",
    "build_classical_classifier",
]
