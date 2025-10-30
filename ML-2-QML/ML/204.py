import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """
    Classical self‑attention module with optional quantum‑derived attention mask.
    """

    def __init__(self, embed_dim: int, dropout: float = 0.1):
        """
        Parameters
        ----------
        embed_dim : int
            Size of the embedding space.
        dropout : float, optional
            Dropout probability applied after the attention sum.
        """
        super().__init__()
        self.embed_dim = embed_dim
        # Linear projection of inputs into embedding space
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """
        Compute self‑attention scores using rotation and entangle parameters.

        Parameters
        ----------
        rotation_params : np.ndarray
            Parameters for the query projection, shape (embed_dim,).
        entangle_params : np.ndarray
            Parameters for the key projection, shape (embed_dim,).
        inputs : np.ndarray
            Input tensor, shape (batch, embed_dim).

        Returns
        -------
        np.ndarray
            Attention output, shape (batch, embed_dim).
        """
        # Project inputs
        x = self.proj(torch.from_numpy(inputs).float())
        # Compute query, key, value
        query = torch.matmul(x, torch.from_numpy(rotation_params.reshape(self.embed_dim, -1)).float())
        key = torch.matmul(x, torch.from_numpy(entangle_params.reshape(self.embed_dim, -1)).float())
        value = x
        # Scaled dot‑product attention
        scores = F.softmax(torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.embed_dim), dim=-1)
        # Weighted sum
        out = torch.matmul(scores, value)
        # Dropout
        out = self.dropout(out)
        return out.detach().numpy()
