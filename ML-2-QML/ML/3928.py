"""Hybrid classical sampler network with embedded fully‑connected layer.

This implementation replaces the minimal seed with a richer architecture:
- A two‑step linear mapping transforms the 2‑dimensional input into 4
  parameters that would feed a quantum sampler.
- A lightweight fully‑connected “quantum‑layer” (here a 4→1 linear
  module followed by tanh) produces a scalar expectation that is
  concatenated with the original input.
- A final classifier maps the 3‑dimensional feature vector to two
  logits, returning a soft‑max probability distribution.

The class is fully PyTorch‑compatible and can be instantiated
directly or wrapped by a higher‑level training loop.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SamplerQNN(nn.Module):
    """
    Classical sampler network that mimics the quantum sampler workflow
    and incorporates a fully‑connected quantum layer concept.
    """

    def __init__(self) -> None:
        super().__init__()
        # Map 2‑dimensional input to 4 parameters that would drive a quantum circuit
        self.input_to_weights = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
        )
        # Fully‑connected quantum‑layer surrogate: 4→1 linear + tanh
        self.fcl = nn.Sequential(
            nn.Linear(4, 1),
            nn.Tanh(),
        )
        # Final classifier: 3‑dimensional feature (2 inputs + 1 expectation) → 2 logits
        self.classifier = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 2).

        Returns
        -------
        torch.Tensor
            Soft‑max probabilities of shape (batch, 2).
        """
        # Generate quantum‑style weights
        weights = self.input_to_weights(x)          # (batch, 4)
        # Compute expectation via the surrogate quantum layer
        expectation = self.fcl(weights).squeeze(-1)  # (batch,)
        # Concatenate original input with expectation
        features = torch.cat([x, expectation.unsqueeze(-1)], dim=-1)  # (batch, 3)
        logits = self.classifier(features)           # (batch, 2)
        return F.softmax(logits, dim=-1)


__all__ = ["SamplerQNN"]
