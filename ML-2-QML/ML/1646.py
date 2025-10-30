import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """
    Multi‑head self‑attention with dropout, residual + layer norm.
    The class keeps the same callable signature as the seed but
    allows configuration of the number of heads and dropout.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        residual: bool = True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.residual = residual

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim) if residual else None

    def forward(
        self,
        inputs: torch.Tensor,
        rotation_params: torch.Tensor,
        entangle_params: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute multi‑head attention. The rotation_params and
        entangle_params are accepted for API parity but ignored,
        mirroring the seed behaviour.
        """
        B, N, _ = inputs.shape
        qkv = self.qkv_proj(inputs)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # reshape for heads
        q = q.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        probs = F.softmax(scores, dim=-1)
        probs = self.dropout(probs)

        attn = torch.matmul(probs, v)
        attn = attn.transpose(1, 2).contiguous().view(B, N, self.embed_dim)
        attn = self.out_proj(attn)

        if self.residual:
            attn = self.norm(attn + inputs)
        return attn

    def get_weights(
        self,
        inputs: torch.Tensor,
        rotation_params: torch.Tensor,
        entangle_params: torch.Tensor,
    ) -> torch.Tensor:
        """Return the attention probabilities for inspection."""
        B, N, _ = inputs.shape
        qkv = self.qkv_proj(inputs)
        q, k, _ = torch.chunk(qkv, 3, dim=-1)
        q = q.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        return F.softmax(scores, dim=-1)

__all__ = ["SelfAttention"]
