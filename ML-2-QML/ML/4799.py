import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------
# Classical Quanvolution components
# ------------------------------------
class QuanvolutionFilter(nn.Module):
    """Classical 2×2 convolution applied to grayscale images."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        return out.view(x.size(0), -1)


class QuanvolutionClassifier(nn.Module):
    """Classifier that uses the `QuanvolutionFilter` as feature extractor."""
    def __init__(self) -> None:
        super().__init__()
        self.filter = QuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.filter(x)
        logits = self.linear(feats)
        return F.log_softmax(logits, dim=-1)


# ------------------------------------
# Classical self‑attention helper
# ------------------------------------
class ClassicalSelfAttention:
    """Simple self‑attention simulation using NumPy tensors."""
    def __init__(self, embed_dim: int) -> None:
        self.embed_dim = embed_dim

    def run(self, rot: np.ndarray, ent: np.ndarray, inp: np.ndarray) -> np.ndarray:
        q = torch.as_tensor(inp @ rot.reshape(self.embed_dim, -1), dtype=torch.float32)
        k = torch.as_tensor(inp @ ent.reshape(self.embed_dim, -1), dtype=torch.float32)
        v = torch.as_tensor(inp, dtype=torch.float32)
        scores = torch.softmax(q @ k.T / math.sqrt(self.embed_dim), dim=-1)
        return (scores @ v).numpy()


# ------------------------------------
# Classical transformer blocks
# ------------------------------------
class MultiHeadAttentionClassical(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_out


class FeedForwardClassical(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlockClassical(nn.Module):
    """Standard transformer block with residual connections."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_out = self.attn(x, mask)
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


# ------------------------------------
# Classical text classifier
# ------------------------------------
class TextClassifier(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_enc = PositionalEncoder(embed_dim)
        self.transformer = nn.Sequential(
            *[TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout) for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_emb(x)
        x = self.pos_enc(tokens)
        x = self.transformer(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)


__all__ = [
    "QuanvolutionFilter",
    "QuanvolutionClassifier",
    "ClassicalSelfAttention",
    "MultiHeadAttentionClassical",
    "FeedForwardClassical",
    "TransformerBlockClassical",
    "PositionalEncoder",
    "TextClassifier",
]
