import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --------------------------------------------------------------------------- #
#  Classical attention primitives
# --------------------------------------------------------------------------- #
class MultiHeadAttentionBase(nn.Module):
    """Base class enforcing dimensionality checks."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError


class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Standard multi‑head attention implemented with PyTorch."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_out


class SelfAttentionClassical(nn.Module):
    """A lightweight self‑attention block used as an alternative to multi‑head attention."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: torch.Tensor) -> torch.Tensor:
        query = torch.matmul(inputs, rotation_params.reshape(self.embed_dim, -1))
        key   = torch.matmul(inputs, entangle_params.reshape(self.embed_dim, -1))
        scores = torch.softmax(query @ key.t() / np.sqrt(self.embed_dim), dim=-1)
        return scores @ inputs


# --------------------------------------------------------------------------- #
#  Classical feed‑forward primitives
# --------------------------------------------------------------------------- #
class FeedForwardBase(nn.Module):
    """Base class for feed‑forward sub‑modules."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class FeedForwardClassical(FeedForwardBase):
    """Two‑layer perceptron feed‑forward network."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# --------------------------------------------------------------------------- #
#  Transformer block primitives
# --------------------------------------------------------------------------- #
class TransformerBlockBase(nn.Module):
    """Base class for transformer blocks."""
    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TransformerBlockClassical(TransformerBlockBase):
    """Classical transformer block with optional self‑attention."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        attention_type: str = "multi",
    ):
        super().__init__(embed_dim, dropout)
        if attention_type == "multi":
            self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        elif attention_type == "self":
            self.attn = SelfAttentionClassical(embed_dim)
        else:
            raise ValueError("attention_type must be'multi' or'self'")
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if isinstance(self.attn, SelfAttentionClassical):
            # In a real setting the user would supply rotation/entangle params.
            # Here we generate dummy parameters for demonstration.
            rotation_params = torch.rand(self.embed_dim * 3).cpu().numpy()
            entangle_params = torch.rand(self.embed_dim - 1).cpu().numpy()
            x = self.attn(rotation_params, entangle_params, x)
        else:
            x = self.attn(x, mask)
        x = self.norm1(x + self.dropout(x))
        x = self.ffn(x)
        return self.norm2(x + self.dropout(x))


# --------------------------------------------------------------------------- #
#  Positional encoding
# --------------------------------------------------------------------------- #
class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-np.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# --------------------------------------------------------------------------- #
#  Hybrid transformer interface
# --------------------------------------------------------------------------- #
class HybridTransformer(nn.Module):
    """Transformer that can be instantiated in a purely classical mode."""
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        attention_type: str = "multi",
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.blocks = nn.ModuleList(
            [
                TransformerBlockClassical(
                    embed_dim, num_heads, ffn_dim, dropout, attention_type
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(
            embed_dim, num_classes if num_classes > 2 else 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        x = self.pos_encoder(x)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)


__all__ = [
    "MultiHeadAttentionBase",
    "MultiHeadAttentionClassical",
    "SelfAttentionClassical",
    "FeedForwardBase",
    "FeedForwardClassical",
    "TransformerBlockBase",
    "TransformerBlockClassical",
    "PositionalEncoder",
    "HybridTransformer",
]
