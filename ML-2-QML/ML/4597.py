"""
Hybrid quantum‑classical classifier – classical side.

This module builds a transformer‑based classifier that optionally
pre‑processes token embeddings through a QCNN‑style feature extractor.
It mirrors the quantum helper interface by exposing a ``build_classifier_circuit``
factory that returns the network, encoding metadata, weight statistics
and observable indices for consistency with the quantum counterpart.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Re‑use the QCNN feature extractor from the seed
try:
    from.QCNN import QCNNModel
except Exception:  # pragma: no cover
    class QCNNModel(nn.Module):
        """Fallback minimal QCNN model used only for local testing."""
        def __init__(self) -> None:
            super().__init__()
            self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
            self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
            self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
            self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
            self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
            self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
            self.head = nn.Linear(4, 1)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            x = self.feature_map(inputs)
            x = self.conv1(x)
            x = self.pool1(x)
            x = self.conv2(x)
            x = self.pool2(x)
            x = self.conv3(x)
            return torch.sigmoid(self.head(x))

        def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
            """Return the intermediate feature vector before the sigmoid."""
            x = self.feature_map(inputs)
            x = self.conv1(x)
            x = self.pool1(x)
            x = self.conv2(x)
            x = self.pool2(x)
            return x

# Minimal transformer blocks – copied from the reference but stripped for clarity
class MultiHeadAttentionClassical(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_output, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_output

class FeedForwardClassical(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlockClassical(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class PositionalEncoder(nn.Module):
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

class HybridClassifier(nn.Module):
    """
    A transformer‑based classifier that can optionally prepend a QCNN feature
    extractor.  The interface matches the quantum helper: ``forward`` accepts
    token indices and returns logits.
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
        use_qcnn: bool = False,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.transformer = nn.Sequential(
            *[TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
              for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

        self.use_qcnn = use_qcnn
        if use_qcnn:
            self.qcnn = QCNNModel()
            # QCNN expects 8‑dimensional inputs; we project embeddings to 8 dims
            self.project_to_qcnn = nn.Linear(embed_dim, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (batch, seq_len)
            Token indices.

        Returns
        -------
        logits : torch.Tensor, shape (batch, num_classes)
        """
        tokens = self.token_embedding(x)           # (B, L, E)
        x = self.pos_encoder(tokens)               # add positional encoding
        x = self.transformer(x)                   # transformer blocks

        # Pool over sequence dimension
        x = x.mean(dim=1)                          # (B, E)

        if self.use_qcnn:
            # Prepare QCNN inputs
            qcnn_input = self.project_to_qcnn(x)    # (B, 8)
            qcnn_features = self.qcnn.forward_features(qcnn_input)
            x = qcnn_features.squeeze(-1)          # (B, 8) -> (B, 8)

        logits = self.classifier(x)
        return logits

def build_classifier_circuit(
    vocab_size: int,
    embed_dim: int,
    num_heads: int,
    num_blocks: int,
    ffn_dim: int,
    num_classes: int,
    dropout: float = 0.1,
    use_qcnn: bool = False,
) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """
    Factory mirroring the quantum helper interface.

    Returns
    -------
    network : nn.Module
        The constructed hybrid classifier.
    encoding : Iterable[int]
        Token indices representation (here simply the range of vocab size).
    weight_sizes : Iterable[int]
        Number of trainable parameters per linear layer for debugging.
    observables : List[int]
        Indices of output logits (0 for binary, 0..num_classes-1 otherwise).
    """
    network = HybridClassifier(
        vocab_size,
        embed_dim,
        num_heads,
        num_blocks,
        ffn_dim,
        num_classes,
        dropout,
        use_qcnn,
    )
    encoding = list(range(vocab_size))
    weight_sizes = [
        p.numel() for p in network.parameters()
    ]
    observables = list(range(num_classes if num_classes > 2 else 1))
    return network, encoding, weight_sizes, observables

__all__ = ["HybridClassifier", "build_classifier_circuit"]
