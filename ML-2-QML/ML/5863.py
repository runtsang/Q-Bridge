"""
SamplerQNNGen263 – Classical deep neural sampler with residual connections and regularisation.
The architecture mirrors the original two‑layer design but adds:
  * Three linear layers with ReLU activations
  * Batch‑normalisation after each hidden layer
  * Dropout for stochastic regularisation
  * A residual skip connection from input to the last hidden layer
  * Softmax output for sampling probabilities
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerQNNGen263(nn.Module):
    """
    A lightweight, fully‑connected sampler network.
    Designed to be drop‑in compatible with the original SamplerQNN interface.
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 8, output_dim: int = 2,
                 dropout_p: float = 0.1) -> None:
        super().__init__()
        # Feature extractor
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        # Residual projection (identity when input_dim == hidden_dim)
        self.residual_proj = nn.Identity() if input_dim == hidden_dim else nn.Linear(input_dim, hidden_dim)
        # Output head
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing a probability distribution over output classes.
        """
        hidden = self.layers(inputs)
        # Add residual connection
        hidden = hidden + self.residual_proj(inputs)
        logits = self.output(hidden)
        return F.softmax(logits, dim=-1)

__all__ = ["SamplerQNNGen263"]
