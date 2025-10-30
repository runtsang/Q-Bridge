"""Hybrid self‑attention that blends classical attention with a neural‑network‑generated parameter layer.

The class mirrors the SelfAttention interface but internally uses a tiny feed‑forward network
(EstimatorQNN style) to produce rotation and entangle parameters for each sample.
This allows the attention mechanism to be conditioned on the input features
and provides a natural bridge to the quantum implementation below.
"""

import numpy as np
import torch
from torch import nn

class HybridSelfAttention:
    """
    Classical self‑attention module.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    hidden_dim : int, optional
        Size of the hidden layer in the parameter‑generation network.
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 16):
        self.embed_dim = embed_dim
        self.param_net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, embed_dim * 2),
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Compute the self‑attention output.

        Parameters
        ----------
        inputs : torch.Tensor
            Input embeddings of shape (batch, embed_dim).

        Returns
        -------
        torch.Tensor
            Attention‑weighted embeddings of shape (batch, embed_dim).
        """
        # Generate rotation and entangle parameters per sample
        params = self.param_net(inputs)  # (batch, embed_dim*2)
        rotation_params = params[:, : self.embed_dim]
        entangle_params = params[:, self.embed_dim :]

        # Element‑wise multiplication to form query and key
        query = inputs * rotation_params
        key = inputs * entangle_params

        # Attention scores per sample: (batch, embed_dim, embed_dim)
        scores = self.softmax(
            torch.einsum("bi,bj->bij", query, key) / np.sqrt(self.embed_dim)
        )

        # Weighted sum over the value (inputs)
        return torch.einsum("bij,bj->bi", scores, inputs)

    def run(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Alias for ``forward`` to match the original SelfAttention API.
        """
        return self.forward(inputs)

__all__ = ["HybridSelfAttention"]
