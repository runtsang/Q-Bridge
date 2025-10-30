"""ML implementation of SamplerQNN with advanced training features.

The class inherits from `torch.nn.Module` and implements:
  * Two hidden layers with ReLU, batch‑norm, and dropout.
  * A probability output via softmax.
  * A `sample` helper that draws samples from the output distribution.
  * Utility to export the learned weights for use in the QML module.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

__all__ = ["SamplerQNN"]


class SamplerQNN(nn.Module):
    """
    Classical neural sampler with dropout, batch‑norm and sampling support.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input vector (default 2).
    hidden_dims : Tuple[int,...]
        Sizes of hidden layers (default (8, 8)).
    dropout_prob : float
        Dropout probability applied after each hidden layer (default 0.2).
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: Tuple[int,...] = (8, 8),
        dropout_prob: float = 0.2,
    ) -> None:
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout_prob))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, input_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning a probability vector.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, input_dim).

        Returns
        -------
        torch.Tensor
            Softmaxed probability distribution of shape (batch, input_dim).
        """
        logits = self.net(x)
        return F.softmax(logits, dim=-1)

    def sample(self, x: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """
        Draw discrete samples from the probability distribution.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, input_dim).
        n_samples : int
            Number of samples to draw per input.

        Returns
        -------
        torch.Tensor
            Sample indices of shape (batch, n_samples).
        """
        probs = self.forward(x)
        return torch.multinomial(probs, n_samples, replacement=True)

    def export_weights(self) -> dict:
        """
        Export the linear layer weights and biases as a dictionary
        compatible with the QML counterpart.

        Returns
        -------
        dict
            Mapping from layer names to ``torch.Tensor`` weights.
        """
        return {
            name: param.detach().cpu().clone()
            for name, param in self.named_parameters()
        }
