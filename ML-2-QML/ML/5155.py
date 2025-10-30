"""Hybrid self‑attention auto‑encoder with convolutional feature extractor and sampler network.

The class mimics the structure of the classical SelfAttention helper while
augmenting it with an auto‑encoder and a simple sampler network.  It can be
used as a drop‑in replacement for the original SelfAttention module in a
classical pipeline.

Typical usage::

    model = HybridSelfAttentionAutoEncoder()
    out = model(inputs)
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class HybridSelfAttentionAutoEncoder(nn.Module):
    """
    A lightweight hybrid encoder that combines:

    1.  A 2×2 convolutional filter (implemented with nn.Conv2d) that
        extracts local features from the input.
    2.  A multi‑head self‑attention block that mixes the extracted
        features with query, key and value projections.
    3.  A fully‑connected auto‑encoder that compresses the attended
        representation into a latent space and reconstructs it.
    4.  A simple sampler network (two linear layers with tanh) that
        interprets the latent representation as a probability vector.
    """

    def __init__(
        self,
        embed_dim: int = 4,
        latent_dim: int = 32,
        hidden_dims: tuple[int, int] = (128, 64),
        kernel_size: int = 2,
        threshold: float = 0.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # 1. Convolutional feature extractor
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            bias=True,
        )
        self.threshold = threshold

        # 2. Self‑attention block
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        # 3. Auto‑encoder
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], embed_dim),
        )

        # 4. Sampler network
        self.sampler = nn.Sequential(
            nn.Linear(latent_dim, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def _conv_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the convolutional filter to a 2×2 patch and return a scalar
        activation.  The input is expected to have shape (N, 1, 2, 2).
        """
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=(2, 3))

    def _attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        A simple dot‑product attention that returns the weighted sum of values.
        """
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scores = F.softmax(q @ k.transpose(-2, -1) / np.sqrt(k.size(-1)), dim=-1)
        return scores @ v

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:

        * Reshape the input to a 2×2 patch if necessary.
        * Extract features with the convolutional filter.
        * Apply self‑attention to produce an attended vector.
        * Encode and decode with the auto‑encoder.
        * Interpret the latent vector with the sampler network.
        """
        # Ensure shape (N, 1, 2, 2)
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(1)
        elif inputs.dim() == 2:
            inputs = inputs.unsqueeze(1).unsqueeze(2).unsqueeze(3)

        feat = self._conv_forward(inputs)  # (N, 1)
        attended = self._attention(feat)   # (N, embed_dim)
        latent = self.encoder(attended)   # (N, latent_dim)
        reconstruction = self.decoder(latent)  # (N, embed_dim)
        probs = F.softmax(self.sampler(latent), dim=-1)  # (N, 2)
        return reconstruction, probs

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the latent representation of the input."""
        feat = self._conv_forward(inputs)
        attended = self._attention(feat)
        return self.encoder(attended)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Reconstruct the attended vector from a latent representation."""
        return self.decoder(latent)

__all__ = ["HybridSelfAttentionAutoEncoder"]
