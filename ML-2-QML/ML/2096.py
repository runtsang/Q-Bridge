import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionEnhanced(nn.Module):
    """
    Multi‑head self‑attention with learnable projections, dropout and optional
    residual connection.  It mirrors the original interface but adds a richer
    representation that can be trained end‑to‑end with standard optimizers.
    """

    def __init__(self, embed_dim: int, num_heads: int = 1,
                 dropout: float = 0.1, residual: bool = True):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.residual = residual
        self.dropout = nn.Dropout(dropout)

        # Projection layers: Q, K, V
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)

        # Output projection
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass for training.
        :param x: Tensor of shape (batch, seq_len, embed_dim)
        :return: Tensor of same shape
        """
        B, N, _ = x.shape
        # Linear projections
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # Reshape for multi‑head
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, D)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot‑product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (B, H, N, D)
        out = out.transpose(1, 2).contiguous().view(B, N, self.embed_dim)
        out = self.W_o(out)

        if self.residual:
            out = out + x
        return out

    def run(self, rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray) -> np.ndarray:
        """
        Run using externally supplied parameter vectors.  The first block of
        rotation_params supplies the weights for W_q, W_k, W_v (flattened and
        reshaped).  The entangle_params supplies the weights for W_o.
        """
        # Set temporary weights
        W_q_shape = self.W_q.weight.shape
        W_k_shape = self.W_k.weight.shape
        W_v_shape = self.W_v.weight.shape
        W_o_shape = self.W_o.weight.shape

        # Expected size of rotation_params: 3 * embed_dim * embed_dim
        assert rotation_params.size == 3 * np.prod(W_q_shape)
        assert entangle_params.size == np.prod(W_o_shape)

        idx = 0
        self.W_q.weight.data = torch.from_numpy(
            rotation_params[idx:idx + np.prod(W_q_shape)]
        ).view(W_q_shape).float()
        idx += np.prod(W_q_shape)
        self.W_k.weight.data = torch.from_numpy(
            rotation_params[idx:idx + np.prod(W_k_shape)]
        ).view(W_k_shape).float()
        idx += np.prod(W_k_shape)
        self.W_v.weight.data = torch.from_numpy(
            rotation_params[idx:idx + np.prod(W_v_shape)]
        ).view(W_v_shape).float()

        self.W_o.weight.data = torch.from_numpy(
            entangle_params.reshape(W_o_shape)
        ).float()

        # Forward pass
        x = torch.from_numpy(inputs).float()
        out = self.forward(x)
        return out.detach().numpy()

__all__ = ["SelfAttentionEnhanced"]
