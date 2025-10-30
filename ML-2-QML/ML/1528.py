import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SelfAttentionLayer(nn.Module):
    """
    Trainable classical self‑attention module.
    Uses multi‑head queries, keys and values with optional dropout.
    Provides a `run` method compatible with the original seed.
    """
    def __init__(self, embed_dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if embed_dim % heads!= 0:
            raise ValueError("embed_dim must be divisible by heads")
        self.embed_dim = embed_dim
        self.heads = heads
        self.head_dim = embed_dim // heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (batch, seq_len, embed_dim)

        Returns
        -------
        out : Tensor of shape (batch, seq_len, embed_dim)
        """
        B, N, _ = x.shape
        q = self.q_proj(x).view(B, N, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, self.embed_dim)
        out = self.out_proj(attn_output)
        return out

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """
        Legacy API that performs the same computation as the original seed.
        The rotation and entangle parameters are interpreted as linear weight matrices.

        Parameters
        ----------
        rotation_params : array-like, shape (embed_dim, embed_dim)
        entangle_params : array-like, shape (embed_dim, embed_dim)
        inputs : array-like, shape (batch, seq_len, embed_dim)

        Returns
        -------
        output : np.ndarray
            Result of the self‑attention operation.
        """
        rot = torch.from_numpy(rotation_params.reshape(self.embed_dim, -1)).float()
        ent = torch.from_numpy(entangle_params.reshape(self.embed_dim, -1)).float()
        inp = torch.from_numpy(inputs).float()

        query = inp @ rot
        key = inp @ ent
        value = inp

        scores = torch.softmax(query @ key.transpose(-2, -1) / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()

    def __repr__(self):
        return f"{self.__class__.__name__}(embed_dim={self.embed_dim}, heads={self.heads})"

__all__ = ["SelfAttentionLayer"]
