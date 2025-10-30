"""Hybrid fully‑connected layer with classical and sampler components.

The class combines a learnable linear layer (mimicking a quantum
fully‑connected layer) with a small neural sampler that outputs a
probability distribution over two classes.  The interface is kept
compatible with the original FCL example: ``run`` returns the
expectation value of the linear layer, while ``sample`` exposes the
sampler network.

The design allows end‑to‑end training of the linear weights together
with the sampler weights, providing a convenient bridge between
classical and quantum experiments.
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F


class HybridFCL(nn.Module):
    """Classical hybrid fully‑connected layer with a sampler network."""

    def __init__(self, n_features: int = 1, sampler_hidden: int = 4) -> None:
        super().__init__()
        # Linear part
        self.linear = nn.Linear(n_features, 1)
        # Sampler network
        self.sampler_net = nn.Sequential(
            nn.Linear(2, sampler_hidden),
            nn.Tanh(),
            nn.Linear(sampler_hidden, 2),
        )

    def run(self, thetas: Iterable[float]) -> torch.Tensor:
        """
        Compute the expectation value of the linear layer.

        Parameters
        ----------
        thetas : Iterable[float]
            Iterable of scalar parameters interpreted as input to the
            linear layer.

        Returns
        -------
        torch.Tensor
            Expectation value as a single‑element tensor.
        """
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.linear(values)).mean(dim=0)
        return expectation

    def sample(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Return a probability distribution from the sampler network.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape ``(batch, 2)`` containing the two input
            parameters for the sampler.

        Returns
        -------
        torch.Tensor
            Softmax probabilities of shape ``(batch, 2)``.
        """
        return F.softmax(self.sampler_net(inputs), dim=-1)


__all__ = ["HybridFCL"]
