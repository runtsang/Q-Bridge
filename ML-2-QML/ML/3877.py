import torch
from torch import nn
import torch.nn.functional as F
import math
from typing import Optional

class MultiHeadAttentionBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError

class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_out

class FeedForwardBase(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class FeedForwardClassical(FeedForwardBase):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlockBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class HybridQCNN(nn.Module):
    """
    Hybrid classical QCNN model that combines convolution-inspired fully‑connected layers
    with transformer blocks for sequence modeling.  It is a drop‑in replacement for the
    original QCNNModel while providing richer representational power.
    """

    def __init__(
        self,
        input_dim: int = 8,
        embed_dim: int = 16,
        num_heads: int = 2,
        num_blocks: int = 2,
        ffn_dim: int = 32,
        num_classes: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # Convolution‑inspired layers
        self.feature_map = nn.Sequential(nn.Linear(input_dim, embed_dim), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(embed_dim, embed_dim // 2), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(embed_dim // 2, embed_dim // 2), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(embed_dim // 2, embed_dim // 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(embed_dim // 4, embed_dim // 4), nn.Tanh())
        self.head = nn.Linear(embed_dim // 4, num_classes)

        # Transformer backbone
        self.pos_encoding = PositionalEncoder(embed_dim // 4)
        self.transformer = nn.Sequential(
            *[
                TransformerBlockClassical(embed_dim // 4, num_heads, ffn_dim, dropout)
                for _ in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expected input shape: (batch, seq_len, input_dim)
        """
        # Convolution‑inspired feature extraction
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)

        # Transformer path
        x = self.pos_encoding(x)
        x = self.transformer(x)

        # Global pooling and classification
        x = x.mean(dim=1)  # (batch, embed_dim//4)
        x = self.dropout(x)
        return torch.sigmoid(self.head(x))

__all__ = ["HybridQCNN"]
