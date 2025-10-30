"""Extended classical sampler network with additional layers and dropout."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SamplerQNNV2(nn.Module):
    """
    A richer classical sampler network.

    Architecture:
    - Two hidden layers with ReLU activations.
    - Dropout for regularisation.
    - Output layer with log_softmax for stable probability estimation.
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
        layers.append(nn.Linear(prev_dim, input_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return log probabilities."""
        logits = self.net(x)
        return F.log_softmax(logits, dim=-1)


__all__ = ["SamplerQNNV2"]
