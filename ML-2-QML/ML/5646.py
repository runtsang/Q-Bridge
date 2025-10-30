"""Enhanced classical self‑attention with residuals, learnable projection, and dropout.

The class mirrors the original API but augments the self‑attention block
with additional trainable parameters and regularisation options,
making it suitable for research experiments that require more expressive
classical attention mechanisms.
"""

import numpy as np
import torch
from typing import Optional

class SelfAttentionEnhanced:
    """Classical self‑attention module with a learnable projection and residual
    connection. The module accepts a rotation matrix ``rotation_params`` and an
    entanglement matrix ``entangle_params``.
    """

    def __init__(
        self,
        embed_dim: int,
        projection: Optional[torch.Tensor] = None,
        residual: bool = True,
        dropout: float = 0.0,
    ):
        """
        Parameters
        ----------
        embed_dim : int
            Dimensionality of the feature space.
        projection : Optional[torch.Tensor], default=None
            Learnable projection matrix of shape (embed_dim, embed_dim).
            If None, an identity matrix is used.
        residual : bool, default=True
            Whether to add a residual connection.
        dropout : float, default=0.0
            Dropout probability applied after the attention output.
        """
        self.embed_dim = embed_dim
        self.residual = residual
        self.dropout = dropout
        if projection is None:
            self.projection = torch.eye(embed_dim, dtype=torch.float32)
        else:
            self.projection = projection.clone().detach().requires_grad_(True)

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Forward pass of the self‑attention block.

        Parameters
        ----------
        rotation_params : np.ndarray
            Parameters for the linear projection of queries and keys.
        entangle_params : np.ndarray
            Parameters for the linear projection of keys.
        inputs : np.ndarray
            Input matrix of shape (batch, embed_dim).

        Returns
        -------
        np.ndarray
            Output of the self‑attention block.
        """
        # Linear projections
        Q = torch.as_tensor(
            inputs @ rotation_params.reshape(self.embed_dim, -1),
            dtype=torch.float32,
        )
        K = torch.as_tensor(
            inputs @ entangle_params.reshape(self.embed_dim, -1),
            dtype=torch.float32,
        )
        V = torch.as_tensor(inputs, dtype=torch.float32)

        # Soft‑max attention
        scores = torch.softmax(Q @ K.T / np.sqrt(self.embed_dim), dim=-1)

        # Weighted sum
        out = scores @ V

        # Projection and residual
        out = out @ self.projection
        if self.residual:
            out = out + inputs

        # Optional dropout
        if self.dropout > 0.0:
            out = torch.nn.functional.dropout(out, p=self.dropout, training=True)

        return out.numpy()

    def __repr__(self):
        return (
            f"<SelfAttentionEnhanced embed_dim={self.embed_dim} "
            f"residual={self.residual} dropout={self.dropout}>"
        )

__all__ = ["SelfAttentionEnhanced"]
