"""
Hybrid sampler – classical side.

The `SamplerQNNGen224` class maps an input vector to a set of
variational parameters that a quantum sampler can consume.
It is a fully classical PyTorch module that can be trained
with standard optimisers.  The network depth and hidden size
are hyper‑parameters that allow scaling from a simple 2→4→2
mapping to deeper, richer models.

Usage
-----
>>> model = SamplerQNNGen224(input_dim=2, hidden_dim=8, depth=3)
>>> input_tensor = torch.tensor([[0.5, -0.2]])
>>> weights = model(input_tensor)          # shape: (batch, weight_dim)
>>> probs = model.probability(weights)     # softmaxed probability vector
>>> samples = model.sample(weights, num_samples=1000)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Tuple


class SamplerQNNGen224(nn.Module):
    """
    Classical feed‑forward network that outputs weights for a quantum sampler.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input feature vector.
    hidden_dim : int
        Size of the hidden layer(s).
    depth : int
        Number of hidden layers; depth > 1 adds more expressive power.
    weight_dim : int
        Number of variational parameters emitted; defaults to
        ``hidden_dim`` to keep the network compact.
    output_dim : int
        Number of output classes for the softmax; typically 2 for binary
        classification or sampling distributions.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 4,
        depth: int = 2,
        weight_dim: int | None = None,
        output_dim: int = 2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.weight_dim = weight_dim or hidden_dim
        self.output_dim = output_dim

        layers: list[torch.nn.Module] = []
        in_dim = input_dim

        # Build a shallow MLP with optional depth
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        # Final layer outputs weight parameters for the quantum circuit
        layers.append(nn.Linear(in_dim, self.weight_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map input features to variational parameters.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, input_dim)

        Returns
        -------
        torch.Tensor
            Shape (batch, weight_dim)
        """
        return self.net(x)

    def probability(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Convert raw weights into a probability distribution using softmax.

        Parameters
        ----------
        weights : torch.Tensor
            Shape (batch, weight_dim)

        Returns
        -------
        torch.Tensor
            Shape (batch, weight_dim) – each row sums to 1.
        """
        return F.softmax(weights, dim=-1)

    def sample(
        self,
        weights: torch.Tensor,
        num_samples: int = 1000,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """
        Draw samples from the probability distribution implied by `weights`.

        Parameters
        ----------
        weights : torch.Tensor
            Shape (batch, weight_dim)
        num_samples : int
            Number of discrete samples to draw per batch element.
        generator : torch.Generator | None
            Optional random generator for reproducibility.

        Returns
        -------
        torch.Tensor
            Shape (batch, num_samples) – indices of sampled categories.
        """
        probs = self.probability(weights)
        return torch.multinomial(probs, num_samples, replacement=True, generator=generator)

    def generate_samples(
        self,
        inputs: torch.Tensor,
        num_samples: int = 224,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """
        Convenience wrapper that runs the network and draws a fixed
        number of samples (default 224) from the resulting distribution.

        Parameters
        ----------
        inputs : torch.Tensor
            Shape (batch, input_dim)
        num_samples : int
            Number of samples to produce per batch element.
        generator : torch.Generator | None
            Optional random generator for reproducibility.

        Returns
        -------
        torch.Tensor
            Shape (batch, num_samples) – sampled indices.
        """
        weights = self(inputs)
        return self.sample(weights, num_samples, generator=generator)


__all__ = ["SamplerQNNGen224"]
