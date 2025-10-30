import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SelfAttentionFusion(nn.Module):
    """
    Classical self‑attention block with trainable linear projections, bias, and dropout.
    The interface mirrors the original seed: run(rotation_params, entangle_params, inputs).
    The rotation_params and entangle_params arguments are ignored in the classical
    implementation but are kept for API compatibility with the quantum version.
    """

    def __init__(self, embed_dim: int, dropout: float = 0.1):
        """
        Parameters
        ----------
        embed_dim : int
            Dimensionality of the input embeddings.
        dropout : float, optional
            Dropout probability applied after the attention output.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout = dropout

        # Trainable linear layers for Q, K, V projections
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=True)

        # Output projection
        self.out_linear = nn.Linear(embed_dim, embed_dim, bias=True)

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
            Unused in the classical implementation; kept for API compatibility.
        entangle_params : np.ndarray
            Unused in the classical implementation; kept for API compatibility.
        inputs : np.ndarray
            Batch of input embeddings of shape (batch_size, embed_dim).

        Returns
        -------
        np.ndarray
            Attention‑weighted embeddings of shape (batch_size, embed_dim).
        """
        x = torch.as_tensor(inputs, dtype=torch.float32, device=self.q_linear.weight.device)

        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)

        scores = torch.softmax(q @ k.transpose(-2, -1) / np.sqrt(self.embed_dim), dim=-1)
        attn_out = scores @ v
        attn_out = self.out_linear(attn_out)
        attn_out = F.dropout(attn_out, p=self.dropout, training=self.training)

        return attn_out.detach().cpu().numpy()

    def validate(
        self,
        quantum_outputs: np.ndarray,
        tolerance: float = 1e-1,
    ) -> bool:
        """
        Simple validation routine that checks whether the classical outputs
        are within a specified tolerance of the quantum outputs.

        Parameters
        ----------
        quantum_outputs : np.ndarray
            Output from the quantum implementation.
        tolerance : float, optional
            Acceptable absolute difference.

        Returns
        -------
        bool
            True if the maximum absolute difference is below the tolerance.
        """
        classical_output = self.run(np.zeros(1), np.zeros(1), quantum_outputs)
        diff = np.abs(classical_output - quantum_outputs)
        return np.all(diff < tolerance)
