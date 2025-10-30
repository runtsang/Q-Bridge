"""Combined classical sampler and autoencoder network.

This module mirrors the original SamplerQNN while adding an
autoencoder backbone.  The network first encodes the 2‑dimensional
input into a latent space, decodes it back, and finally applies a
softmax over the two output logits.  The design is inspired by the
Autoencoder.py seed and the SamplerQNN.py seed, and it can be used
directly in a standard PyTorch training loop.

The public API is a class ``SamplerQNN`` and a factory function
``create_sampler_qnn`` that returns an instance.  The function name
is kept distinct from the class to avoid name clashes while still
preserving the original anchor reference.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SamplerQNN", "create_sampler_qnn"]


class SamplerQNN(nn.Module):
    """Classical sampler network with an autoencoder backbone.

    The architecture follows the classic encoder–decoder pattern
    (see :class:`AutoencoderNet`) but ends with a softmax output
    suitable for probability sampling.  The default configuration
    matches the original 2→4→2 sampler while adding a latent layer
    that can be tuned via ``latent_dim``.
    """

    def __init__(
        self,
        latent_dim: int = 4,
        hidden_dims: tuple[int, int] = (8, 8),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # Encoder: 2 → hidden → latent
        self.encoder = nn.Sequential(
            nn.Linear(2, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], latent_dim),
        )

        # Decoder: latent → hidden → 2
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder, decoder and softmax.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, 2)``.

        Returns
        -------
        torch.Tensor
            Softmaxed output of shape ``(batch, 2)``.
        """
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return F.softmax(recon, dim=-1)


def create_sampler_qnn() -> SamplerQNN:
    """Factory that returns a ready‑to‑use sampler network."""
    return SamplerQNN()
