"""Advanced classical sampler network with dropout, batch‑norm, and log‑probabilities."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SamplerQNNClass(nn.Module):
    """A flexible categorical sampler.

    The network supports an arbitrary number of hidden layers,
    optional batch‑normalisation, dropout and exposes a
    ``log_prob`` helper useful for likelihood‑based training.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: list[int] | tuple[int,...] = (4, 4),
        dropout: float = 0.1,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []

        for idx, h in enumerate(hidden_dims):
            in_dim = input_dim if idx == 0 else hidden_dims[idx - 1]
            layers.append(nn.Linear(in_dim, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dims[-1], input_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return a probability vector over the output categories."""
        logits = self.net(inputs)
        return F.softmax(logits, dim=-1)

    def log_prob(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the log‑probabilities of the categorical distribution."""
        probs = self.forward(inputs)
        return torch.log(probs + 1e-12)

    def sample(
        self, inputs: torch.Tensor, num_samples: int = 1000
    ) -> torch.Tensor:
        """Draw ``num_samples`` draws from the categorical output."""
        probs = self.forward(inputs)
        dist = torch.distributions.Categorical(probs)
        return dist.sample((num_samples,))


def SamplerQNN() -> SamplerQNNClass:
    """Factory that mirrors the original anchor signature."""
    return SamplerQNNClass()


__all__ = ["SamplerQNNClass", "SamplerQNN"]
