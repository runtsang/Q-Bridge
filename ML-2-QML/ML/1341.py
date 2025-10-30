"""Enhanced multi‑head self‑attention implementation using PyTorch.

Features:
- Supports an arbitrary number of attention heads.
- Optional dropout and residual connection.
- Exposes a ``forward`` method compatible with PyTorch modules.
- Provides a simple training example using Adam optimizer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """Multi‑head self‑attention module."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

        # Linear projections for query, key, value
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, embed_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of the same shape as ``x``.
        """
        batch, seq_len, _ = x.size()

        # Linear projections
        q = self.q_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot‑product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)  # (batch, heads, seq_len, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch, seq_len, self.embed_dim)

        # Residual + output projection
        return self.out_proj(attn_output) + x

    @staticmethod
    def example_train_step():
        """Minimal training loop demonstrating usage."""
        batch, seq_len, embed = 2, 5, 16
        model = SelfAttention(embed_dim=embed, num_heads=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        # Dummy data
        inputs = torch.randn(batch, seq_len, embed)
        targets = torch.randn(batch, seq_len, embed)

        model.train()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        return loss.item()
