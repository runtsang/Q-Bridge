"""Hybrid classical self‑attention and classifier module.

The module merges the self‑attention mechanism from the original
SelfAttention seed with a feed‑forward classifier inspired by
QuantumClassifierModel.  The attention parameters are exposed as
`nn.Parameter` so that they can be transferred to a quantum ansatz if
desired.
"""

import torch
import torch.nn as nn
import numpy as np

class HybridSelfAttention(nn.Module):
    """
    Classical self‑attention block followed by a linear classifier.
    """

    def __init__(self, embed_dim: int, num_classes: int = 2):
        super().__init__()
        self.embed_dim = embed_dim

        # Parameters that mirror the quantum rotation and entanglement
        self.rotation_params = nn.Parameter(
            torch.randn(embed_dim * embed_dim, dtype=torch.float32)
        )
        self.entangle_params = nn.Parameter(
            torch.randn(embed_dim * embed_dim, dtype=torch.float32)
        )

        # Classifier head
        self.classifier = nn.Linear(embed_dim, num_classes)

    def attention(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Compute the self‑attention output.

        Parameters
        ----------
        inputs : torch.Tensor
            Input feature matrix of shape (batch, embed_dim).
        """
        query = inputs @ self.rotation_params.view(self.embed_dim, self.embed_dim)
        key = inputs @ self.entangle_params.view(self.embed_dim, self.embed_dim)
        value = inputs

        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return scores @ value

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through attention and classifier.

        Parameters
        ----------
        inputs : torch.Tensor
            Input features of shape (batch, embed_dim).
        """
        attn_out = self.attention(inputs)
        logits = self.classifier(attn_out)
        return logits

__all__ = ["HybridSelfAttention"]
