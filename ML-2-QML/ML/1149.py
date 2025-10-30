import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionModule(nn.Module):
    """
    Multi‑head self‑attention layer compatible with the original interface.
    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    num_heads : int, default 4
        Number of attention heads. Must divide ``embed_dim``.
    dropout : float, default 0.1
        Dropout probability applied to the attention weights.
    """
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Run a single forward pass.
        Parameters
        ----------
        rotation_params : np.ndarray
            Shape ``(embed_dim, embed_dim*3)``.  The first ``embed_dim`` columns
            initialise ``W_q``, the next ``embed_dim`` initialise ``W_k`` and
            the final ``embed_dim`` initialise ``W_v``.
        entangle_params : np.ndarray
            Unused in the classical implementation but kept for API compatibility.
        inputs : np.ndarray
            Shape ``(batch, seq_len, embed_dim)``.
        Returns
        -------
        np.ndarray
            The output of the attention layer, same shape as ``inputs``.
        """
        # Load parameters
        Wq = torch.from_numpy(rotation_params[:, : self.embed_dim]).float()
        Wk = torch.from_numpy(rotation_params[:, self.embed_dim : 2 * self.embed_dim]).float()
        Wv = torch.from_numpy(rotation_params[:, 2 * self.embed_dim :]).float()

        # Apply linear projections
        Q = torch.matmul(inputs, Wq.T)
        K = torch.matmul(inputs, Wk.T)
        V = torch.matmul(inputs, Wv.T)

        # Reshape for multi‑head
        batch, seq_len, _ = Q.shape
        Q = Q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot‑product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)

        # Final projection
        out = self.out_proj(out)

        return out.detach().numpy()

__all__ = ["SelfAttentionModule"]
