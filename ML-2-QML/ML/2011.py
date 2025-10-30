import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    """
    Multi‑head self‑attention with trainable projections.
    rotation_params and entangle_params are optional scalars that modulate
    the learned projections, enabling a direct comparison with the quantum
    version where these arrays control gate angles.
    """
    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor,
                rotation_params: np.ndarray = None,
                entangle_params: np.ndarray = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim).
            rotation_params: Optional 1‑D array of length embed_dim used to scale
                             the query projections.
            entangle_params: Optional 1‑D array of length embed_dim used to scale
                             the key projections.
        Returns:
            output: Tensor of shape (batch, seq_len, embed_dim).
            attn_weights: Tensor of shape (batch, num_heads, seq_len, seq_len).
        """
        B, N, _ = x.size()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        if rotation_params is not None:
            q *= torch.tensor(rotation_params, dtype=q.dtype, device=q.device)
        if entangle_params is not None:
            k *= torch.tensor(entangle_params, dtype=k.dtype, device=k.device)

        # reshape for multi‑head
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # scaled dot‑product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, N, self.embed_dim)
        return self.out_proj(out), attn_weights

def SelfAttention():
    return MultiHeadSelfAttention(embed_dim=4, num_heads=2)

__all__ = ["SelfAttention"]
