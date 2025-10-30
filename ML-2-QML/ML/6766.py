import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --------------------------------------------------------------------------- #
# 1. Classical Convolutional Filter (drop‑in replacement for a quantum filter)
# --------------------------------------------------------------------------- #
class ConvFilter(nn.Module):
    """
    Simple 2‑D convolutional filter that mimics the behaviour of a quantum filter.
    The filter can be applied to any 2‑D patch extracted from token embeddings.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        data : torch.Tensor
            Shape (batch, 1, H, W) or (batch, H, W) where H=W=kernel_size.

        Returns
        -------
        torch.Tensor
            Mean activation after sigmoid thresholding.
        """
        tensor = data if data.ndim == 4 else data.unsqueeze(1)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=[2, 3])


# --------------------------------------------------------------------------- #
# 2. Positional Encoding
# --------------------------------------------------------------------------- #
class PositionalEncoder(nn.Module):
    """
    Sinusoidal positional encoding adapted from the original Transformer paper.
    """
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32) *
            (-math.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# --------------------------------------------------------------------------- #
# 3. Multi‑head Attention (classical)
# --------------------------------------------------------------------------- #
class MultiHeadAttentionBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError


class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_out


# --------------------------------------------------------------------------- #
# 4. Feed‑Forward Network (classical)
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# 5. Transformer Block (classical)
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# 6. Hybrid Text Classifier (classical)
# --------------------------------------------------------------------------- #
class HybridTextClassifier(nn.Module):
    """
    Classical transformer‑based text classifier that optionally embeds a convolutional
    feature extractor. The API mirrors the quantum variant to enable side‑by‑side
    experimentation.
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
        use_conv: bool = False,
        conv_kernel: int = 2,
        conv_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.conv = ConvFilter(kernel_size=conv_kernel, threshold=conv_threshold)
        else:
            self.conv = None

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
              for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            LongTensor of shape (batch, seq_len) with token indices.

        Returns
        -------
        torch.Tensor
            Logits of shape (batch, num_classes) or (batch, 1) for binary.
        """
        tokens = self.token_embedding(x)  # (batch, seq_len, embed_dim)

        if self.use_conv:
            # Reshape each token embedding to a 2‑D patch for convolution.
            batch, seq_len, embed_dim = tokens.shape
            size = int(math.ceil(math.sqrt(embed_dim)))
            pad = size * size - embed_dim
            padded = F.pad(tokens, (0, pad))
            patches = padded.view(batch, seq_len, 1, size, size)  # (batch, seq_len, 1, H, W)

            conv_out = []
            for i in range(seq_len):
                conv_out.append(self.conv(patches[:, i]))
            conv_out = torch.stack(conv_out, dim=1).unsqueeze(-1)  # (batch, seq_len, 1)
            tokens = torch.cat([tokens, conv_out], dim=-1)

        x = self.pos_encoder(tokens)
        x = self.transformer_blocks(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)


__all__ = [
    "ConvFilter",
    "PositionalEncoder",
    "MultiHeadAttentionBase",
    "MultiHeadAttentionClassical",
    "FeedForwardBase",
    "FeedForwardClassical",
    "TransformerBlockBase",
    "TransformerBlockClassical",
    "HybridTextClassifier",
]
