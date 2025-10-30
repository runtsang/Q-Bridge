import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """
    Multi‑head self‑attention with optional projection, bias and layer‑norm.
    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    num_heads : int, default 1
        Number of attention heads.
    dropout : float, default 0.0
        Dropout probability.
    """
    def __init__(self, embed_dim: int, num_heads: int = 1, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear projections for query, key, value
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, rotation_params: np.ndarray | None,
                entangle_params: np.ndarray | None,
                inputs: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        rotation_params : np.ndarray or None
            Matrix of shape (embed_dim, embed_dim) or
            (num_heads, head_dim, head_dim) to override the learned query
            projection weights. If None, the learned weights are used.
        entangle_params : np.ndarray or None
            Matrix of shape (embed_dim, embed_dim) or
            (num_heads, head_dim, head_dim) to override the learned key
            projection weights. If None, the learned weights are used.
        inputs : np.ndarray
            Batched input of shape (batch, seq_len, embed_dim).

        Returns
        -------
        np.ndarray
            Attention output of shape (batch, seq_len, embed_dim).
        """
        # Convert inputs to tensor
        x = torch.as_tensor(inputs, dtype=torch.float32)

        # Override projection weights if parameters are supplied
        if rotation_params is not None:
            w_q = torch.as_tensor(rotation_params.reshape(self.embed_dim, self.embed_dim), dtype=torch.float32)
            self.qkv.weight.data[:self.embed_dim, :] = w_q
        if entangle_params is not None:
            w_k = torch.as_tensor(entangle_params.reshape(self.embed_dim, self.embed_dim), dtype=torch.float32)
            self.qkv.weight.data[self.embed_dim:2*self.embed_dim, :] = w_k

        B, T, _ = x.shape
        qkv = self.qkv(x)  # (B, T, 3*embed_dim)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi‑head attention
        q = q.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        k = k.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot‑product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)  # (B, H, T, D)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.norm(attn_output)

        return attn_output.detach().cpu().numpy()

    def run(self, rotation_params: np.ndarray | None,
            entangle_params: np.ndarray | None,
            inputs: np.ndarray) -> np.ndarray:
        """
        Convenience wrapper that mirrors the original API.
        """
        return self.forward(rotation_params, entangle_params, inputs)

def SelfAttention():
    """
    Factory that returns a ready‑to‑use SelfAttention instance.
    """
    return SelfAttention(embed_dim=4)

__all__ = ["SelfAttention"]
