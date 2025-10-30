import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SelfAttention(nn.Module):
    """
    Multi‑head self‑attention module that mirrors the original interface but
    adds dropout, trainable linear projections and an optional return of
    attention maps.  It can be used as a drop‑in replacement for the
    classical helper while still being fully differentiable.
    """
    def __init__(self, embed_dim: int = 4, n_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        # Linear maps for Q, K, V; weights will be supplied via run()
        self.q_lin = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_lin = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_lin = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.out_lin = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, rotation_params: np.ndarray, entangle_params: np.ndarray,
                inputs: np.ndarray, return_attn: bool = False) -> np.ndarray:
        """
        Parameters
        ----------
        rotation_params : np.ndarray
            Matrix of shape (embed_dim, embed_dim) used as weights for the
            query projection.
        entangle_params : np.ndarray
            Matrix of shape (embed_dim, embed_dim) used as weights for the
            key projection.
        inputs : np.ndarray
            Input tensor of shape (seq_len, embed_dim).
        return_attn : bool
            If True, the method returns a tuple (output, attn_scores).
        """
        # Convert to tensors
        x = torch.as_tensor(inputs, dtype=torch.float32,
                            device=self.q_lin.weight.device)
        # Apply linear projections with supplied weights
        self.q_lin.weight.data = torch.as_tensor(
            rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32,
            device=self.q_lin.weight.device)
        self.k_lin.weight.data = torch.as_tensor(
            entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32,
            device=self.k_lin.weight.device)
        Q = self.q_lin(x)          # (seq_len, embed_dim)
        K = self.k_lin(x)          # (seq_len, embed_dim)
        V = self.v_lin(x)          # (seq_len, embed_dim)

        # Reshape for multi‑head attention
        Q = Q.view(-1, self.n_heads, self.head_dim).transpose(0, 1)   # (n_heads, seq_len, head_dim)
        K = K.view(-1, self.n_heads, self.head_dim).transpose(0, 1)   # (n_heads, seq_len, head_dim)
        V = V.view(-1, self.n_heads, self.head_dim).transpose(0, 1)   # (n_heads, seq_len, head_dim)

        # Scaled dot‑product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)   # (n_heads, seq_len, seq_len)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)                                     # (n_heads, seq_len, head_dim)
        out = out.transpose(0, 1).contiguous().view(-1, self.embed_dim)   # (seq_len, embed_dim)
        out = self.out_lin(out)                                         # (seq_len, embed_dim)

        if return_attn:
            # Return mean attention over heads to mimic original shape
            return out.cpu().numpy(), attn.mean(0).cpu().numpy()
        return out.cpu().numpy()

# Singleton instance for compatibility with the original API
SelfAttention = SelfAttention()

__all__ = ["SelfAttention"]
