"""SamplerQNN for classical probabilistic modeling.

This module defines SamplerQNN, a neural network that maps 2‑dimensional inputs to a
2‑dimensional probability distribution.  Compared to the seed implementation it
adds:

* Two hidden layers with ReLU activation.
* Batch‑normalisation and dropout for regularisation.
* A convenient ``sample`` method that draws from the learned distribution.
* A ``kl_divergence`` helper to evaluate the divergence against a target.

The network is fully torch‑based and can be trained with any PyTorch optimiser.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class SamplerQNN(nn.Module):
    """Extended classical sampler network."""

    def __init__(self, input_dim: int = 2, hidden_dims: tuple[int, int] = (4, 8), output_dim: int = 2, dropout: float = 0.3) -> None:
        """
        Args:
            input_dim: Dimension of the input vector.
            hidden_dims: Tuple of hidden layer sizes.
            output_dim: Number of output classes (probabilities).
            dropout: Dropout probability applied after the second hidden layer.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing a probability distribution over the output classes.

        Args:
            inputs: Tensor of shape (..., input_dim).

        Returns:
            Tensor of shape (..., output_dim) containing class probabilities.
        """
        logits = self.net(inputs)
        return F.softmax(logits, dim=-1)

    def sample(self, inputs: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        Draw samples from the output distribution for each input.

        Args:
            inputs: Tensor of shape (..., input_dim).
            num_samples: Number of samples to draw per input.

        Returns:
            Tensor of shape (..., num_samples) containing sampled class indices.
        """
        probs = self.forward(inputs)
        dist = Categorical(probs)
        return dist.sample((num_samples,)).transpose(0, 1)

    def kl_divergence(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the KL divergence between the model output and a target distribution.

        Args:
            inputs: Tensor of shape (..., input_dim).
            target: Tensor of shape (..., output_dim) containing target probabilities.

        Returns:
            Tensor of shape (...) with the KL divergence for each sample.
        """
        probs = self.forward(inputs)
        eps = 1e-12
        return torch.sum(target * (torch.log(target + eps) - torch.log(probs + eps)), dim=-1)


__all__ = ["SamplerQNN"]
