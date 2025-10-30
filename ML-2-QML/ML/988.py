"""Enhanced self‑attention module using PyTorch."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SelfAttention(nn.Module):
    """
    A multi‑head self‑attention block with dropout and optional
    weight initialization from external parameters.
    """
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                rotation_params: torch.Tensor,
                entangle_params: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (batch, seq_len, embed_dim)
        rotation_params : Tensor of shape (embed_dim, embed_dim)
        entangle_params : Tensor of shape (embed_dim, embed_dim)
        """
        # Initialise linear layers from external parameters
        with torch.no_grad():
            self.q_proj.weight.copy_(rotation_params)
            self.k_proj.weight.copy_(entangle_params)
            self.v_proj.weight.copy_(entangle_params)

        # Linear projections
        q = self.q_proj(x).view(x.size(0), x.size(1), self.num_heads, self.head_dim)
        k = self.k_proj(x).view(x.size(0), x.size(1), self.num_heads, self.head_dim)
        v = self.v_proj(x).view(x.size(0), x.size(1), self.num_heads, self.head_dim)

        # Transpose for attention calculation
        q = q.transpose(1, 2)  # (batch, heads, seq_len, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot‑product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (batch, heads, seq_len, head_dim)
        out = out.transpose(1, 2).reshape(x.size(0), x.size(1), self.embed_dim)
        return self.out_proj(out)

    def run(self, rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray) -> np.ndarray:
        """
        Compatibility wrapper that accepts NumPy arrays and returns NumPy output.
        """
        self.eval()
        with torch.no_grad():
            x = torch.as_tensor(inputs, dtype=torch.float32)
            rot = torch.as_tensor(rotation_params, dtype=torch.float32)
            ent = torch.as_tensor(entangle_params, dtype=torch.float32)
            out = self.forward(x, rot, ent)
            return out.numpy()

__all__ = ["SelfAttention"]
