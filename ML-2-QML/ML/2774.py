import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvFilter(nn.Module):
    """Classical 2×2 convolution that mimics the quanvolution API."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0, bias: bool = True):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations

class MultiHeadAttentionClassical(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq, _ = x.size()
        qkv = self.qkv(x).reshape(batch, seq, 3, self.num_heads, self.embed_dim // self.num_heads)
        q, k, v = qkv.unbind(dim=2)
        scores = torch.einsum('bshd,bshd->bhs', q, k) / math.sqrt(self.embed_dim // self.num_heads)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('bhs,bshd->bshd', attn, v)
        out = out.reshape(batch, seq, self.embed_dim)
        return self.out(out)

class FeedForwardClassical(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlockClassical(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(self.norm1(x))
        x = x + self.dropout(attn_out)
        ffn_out = self.ffn(self.norm2(x))
        return x + self.dropout(ffn_out)

class PositionalEncoder(nn.Module):
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

class HybridTransformer(nn.Module):
    """Classical hybrid transformer that uses a 2×2 Conv2d front‑end."""
    def __init__(self,
                 image_size: int,
                 kernel_size: int = 2,
                 embed_dim: int = 64,
                 num_heads: int = 8,
                 ffn_dim: int = 256,
                 num_blocks: int = 6,
                 num_classes: int = 10,
                 dropout: float = 0.1,
                 conv_threshold: float = 0.0):
        super().__init__()
        self.image_size = image_size
        self.kernel_size = kernel_size
        self.conv = ConvFilter(kernel_size=kernel_size, threshold=conv_threshold)
        self.seq_len = (image_size - kernel_size + 1) ** 2
        self.embed = nn.Linear(1, embed_dim)
        self.pos_enc = PositionalEncoder(embed_dim)
        self.transformer = nn.Sequential(
            *[TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
              for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape (batch, 1, H, W)
        patches = self.conv(x)  # (batch, 1, h', w')
        batch, _, h, w = patches.shape
        patches = patches.view(batch, h * w, 1)
        x = self.embed(patches)
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)

__all__ = [
    "ConvFilter",
    "MultiHeadAttentionClassical",
    "FeedForwardClassical",
    "TransformerBlockClassical",
    "PositionalEncoder",
    "HybridTransformer",
]
