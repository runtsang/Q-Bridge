"""
Hybrid self‑attention module combining classical attention with a quantum‑inspired sampler.

The design integrates the learnable rotation and entanglement layers from the classical attention
seed, while adding an optional torch‑based sampler that can modulate the attention scores.
This allows end‑to‑end training of the entire pipeline with plain PyTorch optimisers.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Optional

class HybridSelfAttention:
    """
    Classical self‑attention block that optionally incorporates a sampler
    to produce probability‑weighted attention scores.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the embedding space.
    sampler : Optional[torch.nn.Module]
        A torch module that outputs a probability distribution over the
        embedding dimension. If supplied, the attention scores are
        element‑wise multiplied by these probabilities before normalisation.
    """

    def __init__(self, embed_dim: int = 4, sampler: Optional[torch.nn.Module] = None):
        self.embed_dim = embed_dim
        self.sampler = sampler

        # Learnable parameters mimicking the rotation and entanglement
        # matrices of the quantum circuit. They are treated as
        # full dense matrices for expressivity.
        self.rotation_params = torch.nn.Parameter(
            torch.randn(embed_dim, embed_dim, dtype=torch.float32)
        )
        self.entangle_params = torch.nn.Parameter(
            torch.randn(embed_dim, embed_dim, dtype=torch.float32)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Compute the attention output.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (batch_size, embed_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, embed_dim).
        """
        # Compute query and key projections
        query = torch.matmul(inputs, self.rotation_params)
        key = torch.matmul(inputs, self.entangle_params)

        # Raw attention scores
        scores = torch.softmax(
            torch.matmul(query, key.T) / np.sqrt(self.embed_dim), dim=-1
        )

        # Modulate with sampler probabilities if provided
        if self.sampler is not None:
            probs = self.sampler(inputs)  # shape: (batch, embed_dim)
            scores = scores * probs
            scores = scores / scores.sum(dim=-1, keepdim=True)

        value = inputs
        return torch.matmul(scores, value)

    def get_attention_weights(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Return the attention weight matrix for inspection.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (batch_size, embed_dim).

        Returns
        -------
        torch.Tensor
            Attention weights of shape (batch_size, embed_dim, embed_dim).
        """
        query = torch.matmul(inputs, self.rotation_params)
        key = torch.matmul(inputs, self.entangle_params)
        scores = torch.softmax(
            torch.matmul(query, key.T) / np.sqrt(self.embed_dim), dim=-1
        )
        if self.sampler is not None:
            probs = self.sampler(inputs)
            scores = scores * probs
            scores = scores / scores.sum(dim=-1, keepdim=True)
        return scores

__all__ = ["HybridSelfAttention"]
