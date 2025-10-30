"""SelfAttentionModule: classical multi‑head attention with optional residual, dropout, and masking."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionModule(nn.Module):
    """
    Classical multi‑head self‑attention block.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    num_heads : int, default=1
        Number of attention heads.
    dropout : float, default=0.1
        Dropout probability applied to attention weights.
    use_residual : bool, default=True
        Whether to add a residual connection from input to output.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 1,
        dropout: float = 0.1,
        use_residual: bool = True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.use_residual = use_residual

        # Linear projections for Q, K, V (no bias to keep symmetry)
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        # Layer‑norm for stabilisation
        self.layernorm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        attn_params: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq, embed_dim).
        attn_params : torch.Tensor
            Element‑wise weighting for the QKV projections, shape
            (batch, seq, 3 * embed_dim).
        mask : torch.Tensor, optional
            Boolean mask broadcastable to (batch, seq, seq) where
            ``True`` indicates positions to mask out.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, seq, embed_dim).
        """
        batch, seq, _ = x.size()

        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # (batch, seq, 3*embed_dim)
        qkv = qkv * attn_params  # element‑wise scaling
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # Reshape for multi‑head
        q = q.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot‑product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32)
        )
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1), float("-inf"))
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values
        weighted = torch.matmul(attn_weights, v)
        weighted = weighted.transpose(1, 2).contiguous().view(batch, seq, self.embed_dim)

        # Output projection + optional residual + layer‑norm
        out = self.out_proj(weighted)
        if self.use_residual:
            out = out + x
        return self.layernorm(out)

__all__ = ["SelfAttentionModule"]
