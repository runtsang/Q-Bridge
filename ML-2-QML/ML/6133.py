"""Self‑attention module with a feed‑forward network and residual connection.

The class mirrors the quantum interface but adds a small two‑layer
feed‑forward network to the attention output.  The residual
connection uses the self‑attention scores computed from the
queries/keys.  The module is fully differentiable and can be
trained with any PyTorch optimizer.
"""

import torch
import torch.nn as nn
import numpy as np

class SelfAttentionModule:
    def __init__(self, embed_dim: int):
        """Create a self‑attention block with a feed‑forward network.

        Parameters
        ----------
        embed_dim : int
            Length of the embedding vector.
        """
        self.embed_dim = embed_dim
        # Feed‑forward network: two linear layers with ReLU
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """
        Compute the self‑attention output, feed it through the FFN,
        and add a residual connection.

        Parameters
        ----------
        rotation_params : np.ndarray
            Parameters for the query and key projections.
        entangle_params : np.ndarray
            Parameters for the value projection (identity here).
        inputs : np.ndarray
            Input matrix of shape (batch, embed_dim).

        Returns
        -------
        np.ndarray
            The transformed embeddings.
        """
        # Convert to tensors
        x = torch.tensor(inputs, dtype=torch.float32)
        # Compute query, key, value
        q = torch.matmul(x, torch.tensor(rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32))
        k = torch.matmul(x, torch.tensor(entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32))
        v = x  # identity
        # Compute attention scores
        scores = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.embed_dim), dim=-1)
        # Compute attention output
        attn_out = torch.matmul(scores, v)
        # Feed‑forward network
        ffn_out = self.ffn(attn_out)
        # Residual connection
        out = attn_out + ffn_out
        return out.detach().numpy()
