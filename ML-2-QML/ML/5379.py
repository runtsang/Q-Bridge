"""Classical sampler network that emulates a quantum sampler.

The network combines:
* a 2‑D convolutional filter (quantum‑like quanvolution),
* a trainable self‑attention block,
* a linear head producing two logits.

The architecture mirrors the quantum sampler but is fully classical and
compatible with PyTorch autograd.  It can be used in place of the
original SamplerQNN module while still offering the same public API.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ClassicalSelfAttention(nn.Module):
    """Trainable self‑attention block inspired by the quantum SelfAttention."""

    def __init__(self, embed_dim: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape (batch, embed_dim).

        Returns
        -------
        torch.Tensor
            Attention‑weighted features of shape (batch, embed_dim).
        """
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scores = F.softmax(q @ k.transpose(-1, -2) / np.sqrt(self.embed_dim), dim=-1)
        return scores @ v


class SamplerQNNGen270(nn.Module):
    """
    Classical sampler network that mimics the quantum SamplerQNN.

    Architecture
    ------------
    * 2‑D convolutional filter (kernel size 2) to emulate quanvolution.
    * Self‑attention block to aggregate spatial features.
    * Linear head producing two logits → softmax probabilities.
    """

    def __init__(self) -> None:
        super().__init__()
        # Convolutional filter (quantum‑like quanvolution)
        self.conv = nn.Conv2d(1, 1, kernel_size=2, bias=True)
        # Self‑attention module
        self.attn = ClassicalSelfAttention(embed_dim=4)
        # Linear head
        self.fc = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 1, H, W).

        Returns
        -------
        torch.Tensor
            Softmax probabilities of shape (batch, 2).
        """
        # Quanvolution step
        conv_out = torch.sigmoid(self.conv(x))
        conv_flat = conv_out.view(conv_out.size(0), -1)

        # Self‑attention over flattened features
        attn_out = self.attn(conv_flat)

        # Concatenate and classify
        combined = torch.cat([conv_flat, attn_out], dim=1)
        logits = self.fc(combined)
        return F.softmax(logits, dim=-1)


def SamplerQNN() -> SamplerQNNGen270:
    """Return an instance of the classical SamplerQNNGen270 network."""
    return SamplerQNNGen270()


__all__ = ["SamplerQNNGen270", "SamplerQNN"]
