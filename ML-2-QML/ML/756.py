import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    """
    Multi‑head self‑attention with learnable rotation and entanglement
    parameters that can be trained end‑to‑end using PyTorch autograd.
    """
    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear projections for query, key, value
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Rotation and entangle parameters (mimic quantum parameters)
        self.rotation_params = nn.Parameter(torch.randn(num_heads, self.head_dim, 3))
        self.entangle_params = nn.Parameter(torch.randn(num_heads, self.head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, embed_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, seq_len, embed_dim).
        """
        batch, seq_len, _ = x.size()

        # Project to query, key, value
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply learned rotation parameters per head
        for h in range(self.num_heads):
            rot = self.rotation_params[h]            # (head_dim, 3)
            q[:, h] = torch.matmul(q[:, h], rot)     # simple linear combination of Euler angles
            k[:, h] = torch.matmul(k[:, h], rot)

        # Scaled dot‑product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)

        # Weighted sum of values
        out = torch.matmul(attn, v)  # (batch, num_heads, seq_len, head_dim)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
        out = self.out_proj(out)
        return out

__all__ = ["SelfAttention"]
