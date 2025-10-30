"""Enhanced classical sampler network with residual connections and dropout.

This module defines a more expressive SamplerQNN class that can be used
in place of the original two‑layer network. The architecture includes
batch normalization, ReLU activations, dropout, and a residual
connection to improve gradient flow. The forward pass returns a
probability distribution over the two output classes via softmax.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SamplerQNN(nn.Module):
    """
    A deepened sampler network for two‑dimensional inputs.

    Parameters
    ----------
    input_dim : int, default 2
        Size of the input feature vector.
    hidden_dim : int, default 8
        Width of the hidden layers.
    output_dim : int, default 2
        Number of output classes.
    dropout : float, default 0.1
        Dropout probability applied after the activation.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 8,
        output_dim: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Residual connection
        self.residual = nn.Linear(input_dim, hidden_dim)

        # Classifier
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Probability distribution over classes of shape
            (batch_size, output_dim).
        """
        # Residual path
        res = self.residual(x)

        # Main path
        out = self.feature_extractor(x)

        # Combine
        out = out + res
        out = F.relu(out)

        logits = self.classifier(out)
        probs = F.softmax(logits, dim=-1)
        return probs

    def parameters(self, recurse: bool = True):
        """
        Return an iterator over the model parameters.
        """
        return super().parameters(recurse=recurse)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(input_dim={self.input_dim}, "
            f"hidden_dim={self.hidden_dim}, output_dim={self.output_dim})"
        )
