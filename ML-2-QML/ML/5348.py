"""Hybrid classical classifier that mirrors the quantum helper interface."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

# Import the individual building blocks from the seed modules
from Conv import Conv
from SelfAttention import SelfAttention
from SamplerQNN import SamplerQNN


class QuantumClassifierModel(nn.Module):
    """
    Classical surrogate of the quantum classifier.

    Architecture:
        1. 2×2 convolution filter (Conv) applied to the first 4 features.
        2. Self‑attention (SelfAttention) that mixes all input features.
        3. Sampler‑style classifier (SamplerQNN) producing a 2‑class probability vector.
    """

    def __init__(
        self,
        num_features: int,
        depth: int,
        kernel_size: int = 2,
        embed_dim: int = 4,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.depth = depth
        self.kernel_size = kernel_size
        self.embed_dim = embed_dim

        # Convolutional filter
        self.conv = Conv()

        # Self‑attention block
        self.attn = SelfAttention()

        # Sampler‑style classifier head
        self.sampler = SamplerQNN()

        # Optional linear layer to match dimensionality
        self.linear = nn.Linear(2, 2)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch, num_features).

        Returns
        -------
        Tensor
            Class probability vector of shape (batch, 2).
        """
        # 1. Convolution on a 2×2 patch of the first 4 features
        patch = x[:, :4].view(-1, self.kernel_size, self.kernel_size)
        conv_out = self.conv.run(patch.numpy())  # scalar per sample

        # 2. Generate attention parameters from the convolution output
        rot_params = np.full((self.embed_dim, self.embed_dim), conv_out)
        ent_params = np.full((self.embed_dim, self.embed_dim), conv_out)

        # 3. Self‑attention forward
        attn_out = self.attn.run(rot_params, ent_params, x.numpy())

        # 4. Sampler classifier
        logits = self.sampler.forward(
            torch.tensor(attn_out, dtype=torch.float32)
        )

        # 5. Final linear (optional)
        return self.linear(logits)


__all__ = ["QuantumClassifierModel"]
