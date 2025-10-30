"""Hybrid classical self‑attention with a fully‑connected head."""
import torch
import torch.nn as nn
import numpy as np
from typing import Iterable

class HybridSelfAttention(nn.Module):
    """
    Classical implementation that mirrors the quantum interface.
    Computes a self‑attention map followed by a learnable linear head.
    """
    def __init__(self, embed_dim: int, n_features: int = 1):
        super().__init__()
        self.embed_dim = embed_dim
        # Linear layers for query, key, value
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)
        # Fully‑connected head
        self.fc = nn.Linear(embed_dim, 1, bias=False)

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray,
            thetas: Iterable[float], inputs: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        rotation_params : np.ndarray
            Parameters for the rotation part of the attention (shape: 3*embed_dim).
        entangle_params : np.ndarray
            Parameters for the entanglement part of the attention (shape: embed_dim-1).
        thetas : Iterable[float]
            Parameters that modulate the fully‑connected head (used as a bias term).
        inputs : np.ndarray
            Input feature matrix (shape: batch, embed_dim).

        Returns
        -------
        np.ndarray
            Output of the fully‑connected head (shape: 1,).
        """
        # Compute query, key, value
        q = self.query(torch.from_numpy(inputs).float())
        k = self.key(torch.from_numpy(inputs).float())
        v = self.value(torch.from_numpy(inputs).float())

        # Self‑attention scores
        scores = torch.softmax(q @ k.transpose(-2, -1) / np.sqrt(self.embed_dim), dim=-1)
        attn_out = scores @ v

        # Aggregate over batch and feed to fully‑connected head
        pooled = attn_out.mean(dim=0, keepdim=True)
        fc_out = torch.tanh(self.fc(pooled))

        # Modulate with external thetas
        theta_sum = torch.tensor(sum(thetas), dtype=torch.float32)
        out = torch.tanh(fc_out + theta_sum)

        return out.detach().numpy()

__all__ = ["HybridSelfAttention"]
