"""Enhanced classical sampler network with sampling and loss utilities."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SamplerQNN(nn.Module):
    """
    Two‑layer neural sampler that outputs a probability vector over two outcomes.
    Extends the original seed by adding a dedicated ``sample`` method and a
    ``cross_entropy`` loss helper for supervised training.
    """

    def __init__(
        self,
        in_features: int = 2,
        hidden_features: int = 8,
        out_features: int = 2,
    ) -> None:
        """
        Parameters
        ----------
        in_features : int
            Dimensionality of the input feature vector.
        hidden_features : int
            Size of the hidden layer.
        out_features : int
            Number of output classes (default 2).
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.Tanh(),
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return a probability distribution over the output classes."""
        logits = self.net(inputs)
        return F.softmax(logits, dim=-1)

    def sample(
        self, inputs: torch.Tensor, num_samples: int = 1
    ) -> torch.Tensor:
        """
        Draw discrete samples from the probability distribution produced by the network.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (*batch, in_features).
        num_samples : int
            Number of samples to draw per input instance.

        Returns
        -------
        torch.Tensor
            Integer samples of shape (*batch, num_samples) with values {0, 1}.
        """
        probs = self.forward(inputs)
        return torch.multinomial(probs, num_samples, replacement=True)

    @staticmethod
    def cross_entropy(
        outputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the cross‑entropy loss between predicted probabilities and one‑hot targets.

        Parameters
        ----------
        outputs : torch.Tensor
            Output probabilities from the network.
        targets : torch.Tensor
            Ground‑truth one‑hot labels.

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        return F.binary_cross_entropy(outputs, targets)


__all__ = ["SamplerQNN"]
