"""Hybrid decoder network for QCNN-based autoencoder."""

from __future__ import annotations

import torch
from torch import nn
from typing import Tuple

class HybridQCNNAutoencoder(nn.Module):
    """
    Classical decoder that reconstructs input data from a latent vector
    produced by a QCNN encoder.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the data to be reconstructed.
    latent_dim : int
        Size of the latent representation produced by the quantum encoder.
    hidden_dims : Tuple[int,...], optional
        Sequence of hidden layer sizes for the decoder MLP.
    dropout : float, optional
        Dropout probability applied after each hidden layer.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: Tuple[int,...] = (64, 128),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = latent_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*layers)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder.

        Parameters
        ----------
        latent : torch.Tensor
            Latent vector of shape (batch, latent_dim).

        Returns
        -------
        torch.Tensor
            Reconstructed data of shape (batch, input_dim).
        """
        return self.decoder(latent)

__all__ = ["HybridQCNNAutoencoder"]
