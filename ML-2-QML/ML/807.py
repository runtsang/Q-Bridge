"""Enhanced classical sampler network with deeper layers, regularisation, and Gumbel‑softmax sampling."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SamplerQNN(nn.Module):
    """
    A flexible, regularised neural sampler.

    Parameters
    ----------
    input_dim : int, default 2
        Dimensionality of the input feature vector.
    hidden_dim : int, default 4
        Size of the hidden linear layers.
    output_dim : int, default 2
        Number of discrete classes to sample from.
    dropout : float, default 0.1
        Dropout probability applied after each hidden layer.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 4,
        output_dim: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning class probabilities.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (..., input_dim).

        Returns
        -------
        torch.Tensor
            Probabilities of shape (..., output_dim).
        """
        logits = self.net(inputs)
        return F.softmax(logits, dim=-1)

    def sample(self, inputs: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
        """
        Sample discrete classes using Gumbel‑softmax trick.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (..., input_dim).
        temperature : float, default 0.5
            Temperature controlling the sharpness of the sample.

        Returns
        -------
        torch.Tensor
            One‑hot encoded samples of shape (..., output_dim).
        """
        probs = self.forward(inputs)
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(probs) + 1e-20) + 1e-20)
        y = (probs + gumbel_noise) / temperature
        y = F.softmax(y, dim=-1)
        return torch.argmax(y, dim=-1)

__all__ = ["SamplerQNN"]
