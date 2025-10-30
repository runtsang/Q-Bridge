"""Unified classical‑only classifier module with optional transformer support.\n\nThe module implements the original `build_classifier_circuit` API while adding a\n`UnifiedClassifier` class that can switch between a shallow feed‑forward network\nand a transformer‑based network.  The design mirrors the structure of the\nreference seed but expands the depth and embedding options.\n"""  

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# 1.  Classical feed‑forward block
# --------------------------------------------------------------------------- #
class _FFNN(nn.Module):
    """Simple fully‑connected network that mimics the original data‑uploading\nfeed‑forward construction.  It exposes `weight_sizes` for API compatibility."""
    def __init__(self, in_features: int, depth: int, hidden_dim: int, out_features: int = 2):
        super().__init__()
        layers: List[nn.Module] = []
        curr = in_features
        for _ in range(depth):
            layers.append(nn.Linear(curr, hidden_dim))
            layers.append(nn.ReLU())
            curr = hidden_dim
        layers.append(nn.Linear(curr, out_features))
        self.net = nn.Sequential(*layers)
        self.weight_sizes = [p.numel() for p in self.parameters()]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --------------------------------------------------------------------------- #
# 2.  Transformer helpers (borrowed from the “QTransformerTorch” seed)
# --------------------------------------------------------------------------- #
class _MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_output, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_output


class _FeedForward(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class _TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = _MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn = _FeedForward(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_out = self.attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class _PositionalEncoder(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class _TextClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = _PositionalEncoder(embed_dim)
        self.transformers = nn.Sequential(
            *[ _TransformerBlock(embed_dim, num_heads, ffn_dim, dropout) for _ in range(num_blocks) ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        x = self.transformers(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)


# --------------------------------------------------------------------------- #
# 3.  Unified classifier that chooses between FFNN and Transformer
# --------------------------------------------------------------------------- #
class UnifiedClassifier(nn.Module):
    """
    Wrapper that exposes either a simple feed‑forward network or a transformer
    pipeline.  The construction is fully deterministic and mirrors the API
    of the original `build_classifier_circuit`.
    """
    def __init__(
        self,
        num_features: int,
        depth: int,
        use_transformer: bool = False,
        num_heads: int = 4,
        ffn_dim: int = 128,
        num_blocks: int = 2,
        num_classes: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if use_transformer:
            vocab_size = num_features + 1  # simple id mapping
            self.model = _TextClassifier(
                vocab_size, num_features, num_heads, num_blocks, ffn_dim, num_classes, dropout
            )
        else:
            self.model = _FFNN(num_features, depth, hidden_dim=num_features, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# --------------------------------------------------------------------------- #
# 4.  Public factory compatible with the original seed
# --------------------------------------------------------------------------- #
def build_classifier_circuit(
    num_features: int,
    depth: int,
    use_transformer: bool = False,
    num_heads: int = 4,
    ffn_dim: int = 128,
    num_blocks: int = 2,
    num_classes: int = 2,
    dropout: float = 0.1,
) -> Tuple[UnifiedClassifier, Iterable[int], List[int], List[int]]:
    """
    Return (model, encoding, weight_sizes, observables).
    *encoding* is a list of indices matching the input feature layout.
    *weight_sizes* is the per‑parameter count list.
    *observables* is a placeholder for compatibility with the quantum API.
    """
    model = UnifiedClassifier(
        num_features,
        depth,
        use_transformer,
        num_heads,
        ffn_dim,
        num_blocks,
        num_classes,
        dropout,
    )
    encoding = list(range(num_features))
    weight_sizes = [p.numel() for p in model.parameters()]
    observables = list(range(num_classes))
    return model, encoding, weight_sizes, observables


__all__ = ["UnifiedClassifier", "build_classifier_circuit"]
