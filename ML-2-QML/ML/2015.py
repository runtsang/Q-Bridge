import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SelfAttention(nn.Module):
    """
    Multi‑head scaled dot‑product attention with optional dropout.
    Mirrors the quantum interface: ``run(rotation_params, entangle_params, inputs)``.
    The ``rotation_params`` and ``entangle_params`` are interpreted as weight
    matrices that can be transferred to the quantum module for a hybrid run.
    """

    def __init__(self, embed_dim: int, num_heads: int = 1, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """
        Standard forward pass: x shape (batch, seq_len, embed_dim).
        """
        batch_size, seq_len, _ = x.size()
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_proj(attn_output)

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: np.ndarray, shots: int = 1024):
        """
        Compatibility layer: interprets the external parameter arrays as
        the weight matrices for the linear projections. This allows a
        shared parameter set with the quantum module.
        """
        # Reshape parameters to square matrices
        W_q = torch.tensor(rotation_params.reshape(self.embed_dim, self.embed_dim), dtype=torch.float32)
        W_k = torch.tensor(entangle_params.reshape(self.embed_dim, self.embed_dim), dtype=torch.float32)
        W_v = torch.tensor(rotation_params.reshape(self.embed_dim, self.embed_dim), dtype=torch.float32)

        # Assign to projection layers
        self.q_proj.weight.data = W_q
        self.k_proj.weight.data = W_k
        self.v_proj.weight.data = W_v

        # Forward
        x = torch.tensor(inputs, dtype=torch.float32)
        return self.forward(x).detach().numpy()

__all__ = ["SelfAttention"]
