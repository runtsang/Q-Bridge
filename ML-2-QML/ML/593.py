import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SelfAttention(nn.Module):
    """
    Multi‑head self‑attention module with trainable projections.
    Supports variable embedding dimension, number of heads, and a classification head.
    """
    def __init__(self, embed_dim: int, num_heads: int = 4, num_classes: int = 10):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor of shape (batch, seq_len, embed_dim)
            Input embeddings.

        Returns
        -------
        logits : torch.Tensor of shape (batch, num_classes)
            Classification logits after attention.
        """
        B, N, _ = x.shape
        qkv = self.qkv_proj(x)  # (B, N, 3*embed_dim)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, D)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, H, N, D)

        attn_scores = torch.einsum('bhqd,bhkd->bhqk', q, k) / np.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, H, Q, K)

        attn_output = torch.einsum('bhqk,bhvd->bhqd', attn_weights, v)  # (B, H, Q, D)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.reshape(B, N, self.embed_dim)  # (B, N, E)

        attn_output = self.out_proj(attn_output)  # (B, N, E)

        pooled = attn_output.mean(dim=1)  # (B, E)
        logits = self.classifier(pooled)  # (B, num_classes)
        return logits
