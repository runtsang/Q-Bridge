import numpy as np
import torch
import math
from typing import Optional

class SelfAttention:
    """
    Hybrid classical self‑attention module with multi‑head capability.
    The run method accepts rotation_params and entangle_params for compatibility
    with the quantum counterpart but uses them as additional projection matrices.
    """
    def __init__(self, embed_dim: int = 64, num_heads: int = 4, dropout: float = 0.1):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        self.dropout = dropout
        # Projection layers
        self.q_proj = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.layer_norm = torch.nn.LayerNorm(embed_dim)
        self.drop = torch.nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        x = x.view(batch, seq_len, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, num_heads, seq_len, head_dim = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(batch, seq_len, num_heads * head_dim)

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """
        Compute multi‑head self‑attention on the inputs.
        rotation_params and entangle_params are used as additional learnable
        weights for the query and key projections to demonstrate compatibility.
        """
        x = torch.as_tensor(inputs, dtype=torch.float32)
        if rotation_params is not None:
            rot = torch.as_tensor(rotation_params.reshape(self.embed_dim, self.embed_dim), dtype=torch.float32)
            self.q_proj.weight.data = self.q_proj.weight.data + rot
        if entangle_params is not None:
            ent = torch.as_tensor(entangle_params.reshape(self.embed_dim, self.embed_dim), dtype=torch.float32)
            self.k_proj.weight.data = self.k_proj.weight.data + ent

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.drop(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = self._combine_heads(context)

        out = self.out_proj(context)
        out = self.drop(out)
        out = self.layer_norm(out + x)  # Residual connection

        return out.detach().numpy()

__all__ = ["SelfAttention"]
