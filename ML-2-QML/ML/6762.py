import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from QCNN import QCNNModel

class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding compatible with transformer input shapes."""
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

class MultiHeadAttentionClassical(nn.Module):
    """Standard multi‑head attention implemented with PyTorch's built‑in module."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_output, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_output

class FeedForwardClassical(nn.Module):
    """Two‑layer feed‑forward network used in transformer blocks."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlockClassical(nn.Module):
    """Transformer block combining classical multi‑head attention and feed‑forward."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
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

class QCNNFeatureExtractor(nn.Module):
    """Wraps the classical QCNNModel to act as a per‑token feature extractor."""
    def __init__(self, embed_dim: int = 8):
        super().__init__()
        if embed_dim!= 8:
            raise ValueError("QCNNModel expects input dimension 8; set embed_dim=8")
        self.qcnn = QCNNModel()
        self.proj = nn.Linear(1, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, embed_dim)
        B, L, D = x.shape
        x_flat = x.view(B * L, D)
        feats = self.qcnn(x_flat)          # (B*L, 1)
        feats = self.proj(feats)           # (B*L, embed_dim)
        return feats.view(B, L, D)

class HybridQuantumTransformer(nn.Module):
    """Hybrid transformer that optionally inserts a classical QCNN feature extractor."""
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 8,
        num_heads: int = 4,
        num_blocks: int = 2,
        ffn_dim: int = 16,
        num_classes: int = 2,
        dropout: float = 0.1,
        use_qcnn: bool = True,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoder(embed_dim)
        self.use_qcnn = use_qcnn
        if use_qcnn:
            self.qcnn_extractor = QCNNFeatureExtractor(embed_dim)
        else:
            self.qcnn_extractor = None
        self.transformer = nn.Sequential(
            *[
                TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
                for _ in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len)
        tokens = self.token_embedding(x)
        x = self.pos_encoding(tokens)
        if self.use_qcnn:
            x = self.qcnn_extractor(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)

__all__ = [
    "PositionalEncoder",
    "MultiHeadAttentionClassical",
    "FeedForwardClassical",
    "TransformerBlockClassical",
    "QCNNFeatureExtractor",
    "HybridQuantumTransformer",
]
