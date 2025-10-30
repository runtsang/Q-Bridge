import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttentionClassical(nn.Module):
    """Standard multi‑head attention implemented purely with PyTorch."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.size()
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, -1).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, -1).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, -1).transpose(1, 2)
        dk = self.embed_dim // self.num_heads
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dk)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
        return self.out_proj(out)

class FeedForwardClassical(nn.Module):
    """Simple two‑layer feed‑forward network."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class HybridTransformerBlock(nn.Module):
    """
    Classical transformer block that optionally supports a quantum gate.
    The `use_quantum` flag is ignored in this module; it is present only for API parity.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        use_quantum: bool = False,
    ):
        super().__init__()
        self.use_quantum = use_quantum
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        if use_quantum:
            raise RuntimeError("Quantum gating is not available in the classical implementation.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))
