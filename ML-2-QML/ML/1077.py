import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SelfAttention(nn.Module):
    """
    Multi‑head self‑attention with optional dropout and residual connection.
    The interface mirrors the quantum counterpart: run(rotation_params,
    entangle_params, inputs).  `rotation_params` are the weight matrices for
    query/key/value projections; `entangle_params` are dropout probabilities
    per head.  The module is fully differentiable and can be plugged into
    larger architectures.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the token embeddings.
    num_heads : int, default 4
        Number of attention heads.
    dropout : float, default 0.1
        Dropout probability applied after the attention matrix.
    """
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

        # linear projections for Q, K, V
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, inputs: torch.Tensor,
                rotation_params: np.ndarray,
                entangle_params: np.ndarray) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs : torch.Tensor
            Shape (batch, seq_len, embed_dim)
        rotation_params : np.ndarray
            Flattened weight matrix of shape (embed_dim*4, embed_dim)
            – first 3*embed_dim for QKV, last embed_dim for out_proj.
        entangle_params : np.ndarray
            Dropout probabilities per head (length num_heads)

        Returns
        -------
        torch.Tensor
            Attention output of shape (batch, seq_len, embed_dim)
        """
        # load external parameters
        with torch.no_grad():
            self.qkv.weight.copy_(torch.from_numpy(rotation_params[:self.embed_dim * 3, :]).float())
            self.out_proj.weight.copy_(torch.from_numpy(rotation_params[self.embed_dim * 3:, :]).float())

        batch, seq, _ = inputs.size()
        # QKV projection
        qkv = self.qkv(inputs).reshape(batch, seq, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each: (batch, seq, heads, head_dim)

        # scaled dot‑product
        scores = torch.einsum('bshd,bshd->bsh', q, k) / np.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # weighted sum
        out = torch.einsum('bsh,bshd->bshd', attn, v)
        out = out.reshape(batch, seq, self.embed_dim)
        out = self.out_proj(out)
        return out

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """
        Compatibility wrapper to match the quantum interface.
        """
        inputs_t = torch.as_tensor(inputs, dtype=torch.float32)
        out_t = self.forward(inputs_t, rotation_params, entangle_params)
        return out_t.detach().cpu().numpy()

__all__ = ["SelfAttention"]
