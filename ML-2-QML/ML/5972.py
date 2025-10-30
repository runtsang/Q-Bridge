"""Hybrid classical graph neural network with sampler integration.

This module extends the original GraphQNN utilities by adding a
parameterised sampler layer at the output.  The architecture can be
instantiated from a list of hidden sizes, and the weights are stored
as learnable PyTorch parameters.  The sampler produces a probability
distribution over two classes, making the network suitable for
classification tasks that benefit from variational sampling.

The class re‑exports the original `random_network`, `feedforward`,
`state_fidelity` and `fidelity_adjacency` helpers so that the
graph‑based utilities remain available.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, List

from.GraphQNN import (
    random_network,
    feedforward,
    state_fidelity,
    fidelity_adjacency,
)

Tensor = torch.Tensor


class HybridGraphQNN(nn.Module):
    """
    Classical hybrid GNN with a sampler output.

    Parameters
    ----------
    qnn_arch : Sequence[int]
        Layer sizes of the underlying feed‑forward network.
    sampler_hidden : int, default 4
        Hidden size of the optional sampler MLP.
    """

    def __init__(self, qnn_arch: Sequence[int], sampler_hidden: int = 4) -> None:
        super().__init__()
        self.qnn_arch = list(qnn_arch)

        # Create learnable weight matrices for each layer
        self.weights = nn.ParameterList(
            [
                nn.Parameter(torch.randn(out, inp))
                for inp, out in zip(self.qnn_arch[:-1], self.qnn_arch[1:])
            ]
        )

        # Sampler MLP that maps the final hidden representation to
        # a two‑dimensional probability vector.
        self.sampler = nn.Sequential(
            nn.Linear(self.qnn_arch[-1], sampler_hidden),
            nn.Tanh(),
            nn.Linear(sampler_hidden, 2),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the classical layers followed by the sampler.

        Parameters
        ----------
        x : Tensor
            Input feature vector of shape (features,).

        Returns
        -------
        Tensor
            Normalised probability vector of shape (2,).
        """
        current = x
        for w in self.weights:
            current = torch.tanh(w @ current)
        probs = F.softmax(self.sampler(current), dim=-1)
        return probs

    @classmethod
    def random_network(cls, qnn_arch: Sequence[int], samples: int = 100) -> tuple:
        """
        Convenience wrapper around the original `random_network` helper
        that returns a fully initialised :class:`HybridGraphQNN` instance
        together with the generated training data.

        Returns
        -------
        tuple
            (arch, weights, training_data, target_weight)
        """
        return random_network(qnn_arch, samples)

    # Re‑export original utilities for convenience
    feedforward = staticmethod(feedforward)
    state_fidelity = staticmethod(state_fidelity)
    fidelity_adjacency = staticmethod(fidelity_adjacency)


__all__ = ["HybridGraphQNN"]
