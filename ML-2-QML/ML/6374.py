"""Hybrid classical model combining QCNN feature extraction with transformer classification."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class QCNNFeatureExtractor(nn.Module):
    """Classical CNN‑like feature map that emulates the QCNN layers described in the original QCNN.py."""
    def __init__(self, input_dim: int = 8, embed_dim: int = 16) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(input_dim, embed_dim), nn.Tanh())
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_map(x)

class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]

class FeedForwardClassical(nn.Module):
    """Standard two‑layer MLP."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class MultiHeadAttentionClassical(nn.Module):
    """Standard multi‑head attention implemented using PyTorch."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_out

class TransformerBlockClassical(nn.Module):
    """Classic transformer block."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_out = self.attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class TextClassifier(nn.Module):
    """Transformer‑based text classifier supporting quantum submodules."""
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 use_quantum: bool = False,
                 use_embeddings: bool = False,
                 vocab_size: int | None = None,
                 **quantum_kwargs) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.use_embeddings = use_embeddings
        if not use_embeddings:
            if vocab_size is None:
                raise ValueError("vocab_size must be provided when use_embeddings is False")
            self.token_embedding = nn.Embedding(vocab_size, embed_dim)
            self.pos_embedding = PositionalEncoder(embed_dim)
        else:
            self.pos_embedding = PositionalEncoder(embed_dim)
        if use_quantum:
            from qml_code import MultiHeadAttentionQuantum, FeedForwardQuantum, TransformerBlockQuantum
            self.blocks = nn.ModuleList([
                TransformerBlockQuantum(
                    embed_dim, num_heads, ffn_dim,
                    n_qubits_transformer=quantum_kwargs.get("n_qubits_transformer", 8),
                    n_qubits_ffn=quantum_kwargs.get("n_qubits_ffn", 8),
                    n_qlayers=quantum_kwargs.get("n_qlayers", 1),
                    q_device=quantum_kwargs.get("q_device"),
                    dropout=dropout
                )
                for _ in range(num_blocks)
            ])
        else:
            self.blocks = nn.ModuleList([
                TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
                for _ in range(num_blocks)
            ])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_embeddings:
            tokens = self.token_embedding(x)
            x = self.pos_embedding(tokens)
        else:
            x = self.pos_embedding(x)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)

class QCNNTransformerHybrid(nn.Module):
    """Hybrid model that first extracts features via a classical QCNN‑style network,
    then feeds the flattened representation into a transformer classifier."""
    def __init__(self,
                 vocab_size: int | None = None,
                 embed_dim: int = 16,
                 num_heads: int = 4,
                 num_blocks: int = 2,
                 ffn_dim: int = 64,
                 num_classes: int = 2,
                 dropout: float = 0.1,
                 use_quantum: bool = False,
                 **quantum_kwargs) -> None:
        super().__init__()
        self.feature_extractor = QCNNFeatureExtractor(input_dim=8, embed_dim=embed_dim)
        self.classifier = TextClassifier(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            ffn_dim=ffn_dim,
            num_classes=num_classes,
            dropout=dropout,
            use_quantum=use_quantum,
            use_embeddings=True,
            vocab_size=vocab_size,
            **quantum_kwargs
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is expected to be of shape (batch, input_dim)
        features = self.feature_extractor(x.float())  # shape: (batch, embed_dim)
        features = features.unsqueeze(1)  # shape: (batch, 1, embed_dim)
        return self.classifier(features)

__all__ = ["QCNNTransformerHybrid", "QCNNFeatureExtractor", "TextClassifier", "TransformerBlockClassical", "PositionalEncoder"]
