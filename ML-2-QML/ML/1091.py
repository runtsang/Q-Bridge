import torch
import torch.nn as nn
import numpy as np

class HybridSelfAttention(nn.Module):
    """
    Multi‑head self‑attention block with residual connection and layer‑norm.
    The weights of the linear projections are supplied via the *rotation_params*
    argument, allowing the same interface used by the quantum counterpart.
    """
    def __init__(self, embed_dim: int, num_heads: int = 1, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(embed_dim)

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray,
            inputs: np.ndarray) -> np.ndarray:
        """
        Compute the self‑attention output.
        Parameters
        ----------
        rotation_params : np.ndarray
            Flat array of length 3*embed_dim*embed_dim containing the weights
            for the Q, K and V projections in that order.
        entangle_params : np.ndarray
            Unused placeholder to keep the signature compatible with the
            quantum implementation.
        inputs : np.ndarray
            Input tensor of shape (batch, embed_dim).
        Returns
        -------
        np.ndarray
            The attended representation of shape (batch, embed_dim).
        """
        # Load weights from rotation_params
        w_q = rotation_params[:self.embed_dim * self.embed_dim].reshape(self.embed_dim, self.embed_dim)
        w_k = rotation_params[self.embed_dim * self.embed_dim:
                              2 * self.embed_dim * self.embed_dim].reshape(self.embed_dim, self.embed_dim)
        w_v = rotation_params[2 * self.embed_dim * self.embed_dim:].reshape(self.embed_dim, self.embed_dim)

        self.q.weight = nn.Parameter(torch.tensor(w_q, dtype=torch.float32))
        self.k.weight = nn.Parameter(torch.tensor(w_k, dtype=torch.float32))
        self.v.weight = nn.Parameter(torch.tensor(w_v, dtype=torch.float32))

        x = torch.as_tensor(inputs, dtype=torch.float32)

        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)

        scores = torch.softmax(Q @ K.transpose(-2, -1) / np.sqrt(self.head_dim), dim=-1)
        attn_output = torch.matmul(scores, V)

        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout(attn_output)

        # Residual + LayerNorm
        return self.layernorm(x + attn_output).detach().numpy()

__all__ = ["HybridSelfAttention"]
