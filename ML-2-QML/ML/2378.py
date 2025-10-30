"""Hybrid fully‑connected layer with a classical sampler.

The module mirrors the classical FCL and SamplerQNN seeds, merging their
behaviour into a single PyTorch nn.Module.  It exposes a `forward` method
that takes a vector of thetas (for the linear transformation) and a batch
of two‑dimensional inputs (for the sampler).  The linear output biases
the first input component before passing it through a softmax‑activated
sampler network.  A `run` helper returns the linear expectation value
for debugging or hybrid training purposes."""
from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F


def HybridFCLSampler() -> nn.Module:
    class _HybridFCLSampler(nn.Module):
        def __init__(self, n_features: int = 1) -> None:
            super().__init__()
            # Classic fully‑connected mapping
            self.linear = nn.Linear(n_features, 1)
            # Classic sampler network
            self.sampler = nn.Sequential(
                nn.Linear(2, 4),
                nn.Tanh(),
                nn.Linear(4, 2),
            )

        def forward(
            self, thetas: torch.Tensor, inputs: torch.Tensor
        ) -> torch.Tensor:
            """
            Parameters
            ----------
            thetas : torch.Tensor
                Shape (n_features,) – parameters for the linear layer.
            inputs : torch.Tensor
                Shape (batch, 2) – inputs for the sampler network.

            Returns
            -------
            torch.Tensor
                Softmax probabilities of shape (batch, 2).
            """
            # Linear expectation
            expectation = torch.tanh(self.linear(thetas))
            # Bias the first input component with the expectation
            biased_inputs = inputs.clone()
            biased_inputs[:, 0] += expectation.squeeze()
            # Sampler output
            probs = F.softmax(self.sampler(biased_inputs), dim=-1)
            return probs

        def run(self, thetas: torch.Tensor) -> float:
            """
            Return the linear expectation value for a given theta vector.
            """
            return self.linear(thetas).item()

    return _HybridFCLSampler()


__all__ = ["HybridFCLSampler"]
