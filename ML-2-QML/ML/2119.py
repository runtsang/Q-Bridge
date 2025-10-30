import numpy as np
import torch
import torch.nn.functional as F

class SelfAttentionModel:
    """Multi‑head self‑attention with optional dropout."""
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = torch.nn.Dropout(dropout)

        # Linear projections
        self.q_linear = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_linear = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_linear = torch.nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, embed_dim)
        Returns:
            Tensor of shape (batch, seq_len, embed_dim)
        """
        batch, seq, _ = x.size()

        # Linear projections
        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)

        # Reshape for multi‑head attention
        Q = Q.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot‑product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(batch, seq, self.embed_dim)
        out = self.out_linear(out)
        return out

    def run(self, inputs: np.ndarray) -> np.ndarray:
        """
        Convenience wrapper for numpy input.
        """
        x = torch.as_tensor(inputs, dtype=torch.float32)
        out = self.forward(x)
        return out.numpy()

__all__ = ["SelfAttentionModel"]
