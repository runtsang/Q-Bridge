"""Pure PyTorch implementation of a hybrid transformer classifier.
It optionally supports a quantum transformer branch via a boolean flag, but
remains fully classical to satisfy the ML output constraint."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Local convolutional filter (classical)
def Conv():
    """Return a callable object that emulates the quantum filter with PyTorch ops."""
    class ConvFilter(nn.Module):
        def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
            super().__init__()
            self.kernel_size = kernel_size
            self.threshold = threshold
            # 1‑channel in/out to emulate a single‑channel filter
            self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True, stride=kernel_size)

        def forward(self, data: torch.Tensor) -> torch.Tensor:
            # data shape: (batch, 1, H, W)
            logits = self.conv(data)
            activations = torch.sigmoid(logits - self.threshold)
            return activations
    return ConvFilter()

# Classical transformer primitives
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_output, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_output

class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn = FeedForward(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_out = self.attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-torch.log(torch.tensor(10000.0)) / embed_dim)
        )
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]

class HybridTransformerClassifier(nn.Module):
    """Transformer‑based text/image classifier with a quantum‑aware API.
    Parameters
    ----------
    vocab_size : int
        Size of the token vocabulary (for text) or 1 for image data.
    embed_dim : int
        Dimensionality of token embeddings.
    num_heads : int
        Number of attention heads.
    num_blocks : int
        Number of transformer blocks.
    ffn_dim : int
        Hidden dimension in the feed‑forward network.
    num_classes : int
        Number of target classes.
    dropout : float, default=0.1
        Dropout probability.
    use_quantum : bool, default=False
        If True, the model will instantiate quantum transformer blocks
        in the QML module only. The ML module ignores this flag.
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
        use_quantum: bool = False,
    ) -> None:
        super().__init__()
        self.use_quantum = use_quantum
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        # Convolutional front‑end (classical)
        self.conv = Conv()
        # Transformer stack
        self.transformers = nn.Sequential(
            *[
                TransformerBlock(embed_dim, num_heads, ffn_dim, dropout)
                for _ in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If input is image data (shape: batch, 1, H, W), apply conv
        if x.ndim == 4:
            x = self.conv(x)  # shape: batch, 1, H', W'
            # flatten patches into tokens
            batch, c, h, w = x.shape
            x = x.view(batch, -1, c)  # tokens of size 1
        # Token embedding
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        x = self.transformers(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)

__all__ = [
    "HybridTransformerClassifier",
]
