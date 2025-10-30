import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention__gen364(nn.Module):
    """
    Multi‑head self‑attention module with trainable linear projections and dropout.
    Mirrors the interface of the seed but adds depth and regularisation.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (batch, seq_len, embed_dim)
            Input embeddings.
        mask : Tensor of shape (batch, seq_len, seq_len) or None
            Optional mask where positions with value 0 are ignored.

        Returns
        -------
        Tensor of shape (batch, seq_len, embed_dim)
            Output of the attention block.
        """
        batch, seq_len, _ = x.size()
        qkv = self.qkv_proj(x).reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq, dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # scaled dot‑product
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.head_dim**0.5
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch, seq_len, self.embed_dim)
        return self.out_proj(out)

__all__ = ["SelfAttention__gen364"]
