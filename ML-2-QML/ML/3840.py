import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvFilter(nn.Module):
    """A lightweight 2‑D convolutional filter that emulates a quantum Quanv layer."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        logits = self.conv(data)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=(2,3))

class MultiHeadAttentionBase(nn.Module):
    """Shared logic for attention layers."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.size()
        return x.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, heads, seq, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch, seq, heads * d_k)

    def forward(self, *args, **kwargs) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Standard multi‑head attention implemented with PyTorch."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq, _ = x.size()
        k = self._split_heads(self.k_proj(x))
        q = self._split_heads(self.q_proj(x))
        v = self._split_heads(self.v_proj(x))

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = self._merge_heads(out)
        return self.out_proj(out)

class FeedForwardBase(nn.Module):
    """Base for feed‑forward networks."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

class FeedForwardClassical(FeedForwardBase):
    """Two‑layer MLP used after attention."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.fc1 = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.fc2 = nn.Linear(ffn_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.relu(self.fc1(x))))

class TransformerBlockBase(nn.Module):
    """Layer that stacks attention and feed‑forward."""
    def __init__(self, embed_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

class TransformerBlockClassical(TransformerBlockBase):
    """Pure‑classical transformer block."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, dropout)
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

class HybridTransformerClassifier(nn.Module):
    """
    A transformer‑based classifier that optionally replaces the tokenisation
    stage with a lightweight quantum‑friendly convolutional filter.
    """
    def __init__(
        self,
        vocab_size: int | None = None,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_blocks: int = 6,
        ffn_dim: int = 512,
        num_classes: int = 2,
        dropout: float = 0.1,
        use_conv: bool = False,
        patch_size: int = 2,
        threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            if vocab_size is not None:
                raise ValueError("When use_conv=True, vocab_size should be None")
            self.conv_filter = ConvFilter(kernel_size=patch_size, threshold=threshold)
            self.token_embed = nn.Linear(1, embed_dim, bias=True)
        else:
            if vocab_size is None:
                raise ValueError("vocab_size must be provided when use_conv=False")
            self.token_embed = nn.Embedding(vocab_size, embed_dim)

        self.pos_embed = PositionalEncoder(embed_dim)
        blocks = [
            TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_blocks)
        ]
        self.transformer = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len) if use_conv=False
           (batch, H, W) if use_conv=True
        """
        if self.use_conv:
            patches = x.unfold(1, self.conv_filter.kernel_size, self.conv_filter.kernel_size)\
                       .unfold(2, self.conv_filter.kernel_size, self.conv_filter.kernel_size)
            batch, num_patches_h, num_patches_w, k, k2 = patches.shape
            num_patches = num_patches_h * num_patches_w
            patches = patches.reshape(batch, num_patches, k, k)
            patch_vals = torch.zeros(batch, num_patches, device=x.device)
            for i in range(batch):
                for j in range(num_patches):
                    patch = patches[i, j]
                    val = self.conv_filter(patch.unsqueeze(0))
                    patch_vals[i, j] = val
            tokens = self.token_embed(patch_vals.unsqueeze(-1))
        else:
            tokens = self.token_embed(x)
        tokens = self.pos_embed(tokens)
        out = self.transformer(tokens)
        out = out.mean(dim=1)
        out = self.dropout(out)
        return self.classifier(out)

__all__ = [
    "ConvFilter",
    "MultiHeadAttentionBase",
    "MultiHeadAttentionClassical",
    "FeedForwardBase",
    "FeedForwardClassical",
    "TransformerBlockBase",
    "TransformerBlockClassical",
    "PositionalEncoder",
    "HybridTransformerClassifier",
]
