from __future__ import annotations
import torch
from torch import nn
import numpy as np

class HybridSelfAttention(nn.Module):
    """
    Classical hybrid self‑attention that combines a standard
    attention mechanism with a lightweight fully‑connected
    sub‑module.  The attention weights are computed from
    rotation_params and entangle_params, mirroring the
    interface of the quantum reference, while the value
    projection uses a learnable fully‑connected layer.
    """
    def __init__(self, embed_dim: int = 4, fc_features: int = 1):
        super().__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, inputs: np.ndarray,
                rotation_params: np.ndarray,
                entangle_params: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        inputs : np.ndarray
            Input matrix of shape (batch, embed_dim).
        rotation_params : np.ndarray
            Parameters used to modulate the query projection.
        entangle_params : np.ndarray
            Parameters used to modulate the key projection.
        Returns
        -------
        np.ndarray
            Refined attention output of shape (batch, 1).
        """
        q = self.query(torch.as_tensor(inputs, dtype=torch.float32))
        k = self.key(torch.as_tensor(inputs, dtype=torch.float32))

        q = q * torch.as_tensor(rotation_params.reshape(1, -1), dtype=torch.float32)
        k = k * torch.as_tensor(entangle_params.reshape(1, -1), dtype=torch.float32)

        scores = torch.softmax(q @ k.T / np.sqrt(self.embed_dim), dim=-1)
        v = self.value(torch.as_tensor(inputs, dtype=torch.float32))
        attn_out = scores @ v
        out = self.fc(attn_out)
        return out.detach().numpy()

    run = forward

__all__ = ["HybridSelfAttention"]
