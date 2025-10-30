from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SamplerQNN"]


class SamplerModule(nn.Module):
    """
    A lightweight but expressive sampler network.
    It maps a 2‑dimensional input to a categorical distribution over 2 classes.
    The architecture consists of two hidden layers with ReLU activations, dropout
    for regularisation, and a final softmax output.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: list[int] | tuple[int,...] = (8, 8),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing a probability distribution.
        """
        return F.softmax(self.net(x), dim=-1)

    def sample(self, x: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """
        Draw samples from the categorical distribution defined by the network output.
        Returns a tensor of shape (n_samples, batch_size) with integer labels.
        """
        probs = self.forward(x)
        dist = torch.distributions.Categorical(probs)
        return dist.sample((n_samples,)).transpose(0, 1)


def SamplerQNN() -> SamplerModule:
    """
    Factory returning an instance of :class:`SamplerModule`.
    """
    return SamplerModule()
