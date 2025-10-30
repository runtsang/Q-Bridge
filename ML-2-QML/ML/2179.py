import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """
    Enhanced classical self‑attention module.
    Supports batched inputs, optional bias, dropout and a causal mask.
    """
    def __init__(self, embed_dim: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout)
        self.q_lin = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_lin = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_lin = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        inputs: torch.Tensor,
        rotation_params: torch.Tensor,
        entangle_params: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute self‑attention.

        Parameters
        ----------
        inputs : torch.Tensor
            Input embeddings of shape (batch, seq_len, embed_dim).
        rotation_params : torch.Tensor
            Weight matrix for Q projection of shape (embed_dim, embed_dim).
        entangle_params : torch.Tensor
            Weight matrix for K projection of shape (embed_dim, embed_dim).
        mask : torch.Tensor or None
            Optional mask of shape (seq_len, seq_len) where True indicates
            positions that should be masked out.

        Returns
        -------
        torch.Tensor
            Attention output of shape (batch, seq_len, embed_dim).
        """
        # Apply user‑defined projection matrices. They are expected to be
        # compatible with the linear layers above and are only used to keep
        # the original seed signature.
        q = self.q_lin(inputs) @ rotation_params.T
        k = self.k_lin(inputs) @ entangle_params.T
        v = self.v_lin(inputs)

        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.embed_dim)

        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        probs = F.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        return torch.matmul(probs, v)

    # Preserve the original seed interface
    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        inputs_t = torch.as_tensor(inputs, dtype=torch.float32)
        rotation_t = torch.as_tensor(rotation_params, dtype=torch.float32)
        entangle_t = torch.as_tensor(entangle_params, dtype=torch.float32)
        mask_t = torch.as_tensor(mask, dtype=torch.bool) if mask is not None else None
        out = self.forward(inputs_t, rotation_t, entangle_t, mask_t)
        return out.detach().numpy()

__all__ = ["SelfAttention"]
