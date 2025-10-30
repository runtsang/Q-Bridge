"""Enhanced classical sampler network with deeper architecture and sampling utilities."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, List, Optional

class SamplerModule(nn.Module):
    """
    A flexible softmax sampler built on a multi‑layer perceptron.

    Parameters
    ----------
    input_dim : int, default 2
        Dimensionality of the input vector.
    hidden_dims : Sequence[int], default (4, 4)
        Width of each hidden layer.
    output_dim : int, default 2
        Number of classes.
    dropout : float, default 0.1
        Drop‑out probability applied after each hidden layer.
    """
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: Sequence[int] = (4, 4),
        output_dim: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return class probabilities."""
        logits = self.net(inputs)
        return F.softmax(logits, dim=-1)

    def sample_from_probs(
        self,
        probs: torch.Tensor,
        num_samples: int = 1,
        replacement: bool = True,
    ) -> torch.Tensor:
        """
        Sample discrete labels from a probability vector.

        Parameters
        ----------
        probs : torch.Tensor
            Shape (..., num_classes). Must sum to one along the last dimension.
        num_samples : int
            Number of samples to draw per probability vector.
        replacement : bool
            Whether samples are drawn with replacement.
        """
        return torch.multinomial(probs, num_samples, replacement=replacement)

    def freeze(self) -> None:
        """Freeze all parameters to prevent gradient updates."""
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze all parameters."""
        for p in self.parameters():
            p.requires_grad = True

def SamplerQNN() -> SamplerModule:
    """
    Factory that returns a ready‑to‑train sampler module.

    The default architecture mirrors the original 2‑layer network but adds
    batch‑normalisation, dropout and an extra hidden layer for better
    generalisation.
    """
    return SamplerModule()

__all__ = ["SamplerQNN"]
