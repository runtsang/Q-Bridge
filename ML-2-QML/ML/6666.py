import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """
    Classical self‑attention module that learns a linear projection and
    implements multi‑head attention. The module can be used as a drop‑in
    replacement for the original seed, but it now contains a learnable
    projection matrix and a dropout layer for regularisation.
    """

    def __init__(self, embed_dim: int, num_heads: int = 1, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads

        # Linear projections for query, key, value
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, embed_dim)
        Returns:
            Tensor of shape (batch_size, seq_len, embed_dim)
        """
        B, N, _ = x.shape
        qkv = self.qkv_proj(x)  # (B, N, 3*embed_dim)
        q, k, v = qkv.chunk(3, dim=-1)  # each (B, N, embed_dim)

        # Reshape for multi‑head
        q = q.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, heads, N, head_dim)
        k = k.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot‑product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        attn_output = torch.matmul(attn_probs, v)  # (B, heads, N, head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(B, N, self.embed_dim)

        out = self.out_proj(attn_output)
        return out

    def run(self, rotation_params: torch.Tensor, entangle_params: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        """
        Compatibility wrapper that mimics the original seed interface.
        The rotation_params and entangle_params are ignored in this
        classical implementation but are kept for API compatibility.
        """
        return self.forward(inputs)
