import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Iterable, Tuple, List

# ---------- Transformer primitives (from QTransformerTorch) ----------
class MultiHeadAttentionBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        return torch.matmul(scores, v), scores

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        q = self.separate_heads(x)
        k = self.separate_heads(x)
        v = self.separate_heads(x)
        attn_out, _ = self.attention(q, k, v, mask)
        return attn_out.transpose(1, 2).contiguous().view(x.size(0), -1, self.embed_dim)

class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__(embed_dim, num_heads, dropout)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.combine_heads = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)
        attn_out = super().forward(torch.cat([q, k, v], dim=2), mask)
        return self.combine_heads(attn_out)

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
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

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

# ---------- Classical quanvolution filter ----------
class QuanvolutionFilter(nn.Module):
    """Classical 2‑D convolution filter with a tunable threshold."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.view(x.size(0), -1)

# ---------- Hybrid classifier ----------
class QuanvolutionClassifier(nn.Module):
    """
    Hybrid classifier that can operate in a purely classical mode or augment
    the feature extractor with a transformer stack for richer representations.
    """
    def __init__(
        self,
        use_transformer: bool = False,
        num_classes: int = 10,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_blocks: int = 2,
        ffn_dim: int = 256,
        threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.filter = QuanvolutionFilter(threshold=threshold)
        if use_transformer:
            # Flatten filter output to sequence length 14*14
            seq_len = 14 * 14
            self.proj = nn.Linear(seq_len, embed_dim)
            self.transformer = nn.Sequential(
                *[TransformerBlockClassical(embed_dim, num_heads, ffn_dim) for _ in range(num_blocks)]
            )
            self.classifier = nn.Linear(embed_dim, num_classes)
        else:
            self.linear = nn.Linear(1 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.filter(x)
        if hasattr(self, "proj"):
            seq = self.proj(features).unsqueeze(1)  # batch, seq_len, embed_dim
            x = self.transformer(seq).squeeze(1)
            logits = self.classifier(x)
        else:
            logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

# ---------- Classical classifier circuit builder ----------
def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """
    Construct a feed‑forward classifier that mirrors the quantum circuit interface.
    Returns the network, an encoding mask, weight sizes, and observable indices.
    """
    layers = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.append(linear)
        layers.append(nn.ReLU())
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables = list(range(2))
    return network, encoding, weight_sizes, observables

__all__ = [
    "QuanvolutionFilter",
    "QuanvolutionClassifier",
    "build_classifier_circuit",
    "TransformerBlockClassical",
    "PositionalEncoder",
]
