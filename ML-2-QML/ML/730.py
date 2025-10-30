"""
SamplerQNN__gen406 – Classical sampler with configurable architecture.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Iterable, Tuple

__all__ = ["SamplerQNN"]


def SamplerQNN(
    input_dim: int = 2,
    hidden_dims: Iterable[int] = (4,),
    output_dim: int = 2,
    activation: Callable[[torch.Tensor], torch.Tensor] = nn.Tanh,
    seed: int | None = None,
) -> nn.Module:
    """
    Factory that returns a neural sampler module.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input feature vector.
    hidden_dims : Iterable[int]
        Sizes of hidden layers; can be an empty tuple for a linear model.
    output_dim : int
        Dimensionality of the output probability vector.
    activation : Callable
        Non‑linear activation applied after each hidden layer.
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    nn.Module
        A module that maps an input tensor to a probability vector via
        softmax.  It also exposes a ``sample`` method that draws samples
        from the categorical distribution.
    """
    if seed is not None:
        torch.manual_seed(seed)

    layers: list[nn.Module] = []

    prev_dim = input_dim
    for h_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, h_dim))
        layers.append(activation())
        prev_dim = h_dim

    layers.append(nn.Linear(prev_dim, output_dim))

    net = nn.Sequential(*layers)

    class SamplerModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = net

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return F.softmax(self.net(inputs), dim=-1)

        def sample(
            self,
            inputs: torch.Tensor,
            n_samples: int = 1024,
            *,
            generator: torch.Generator | None = None,
        ) -> torch.Tensor:
            """
            Draw samples from the categorical distribution defined by the
            network's output.

            Parameters
            ----------
            inputs : torch.Tensor
                Batch of input vectors, shape ``(batch, input_dim)``.
            n_samples : int
                Number of categorical samples per input.
            generator : torch.Generator | None
                Optional generator for reproducibility.

            Returns
            -------
            torch.Tensor
                Samples of shape ``(batch, n_samples)`` with integer labels.
            """
            probs = self.forward(inputs)
            return torch.multinomial(
                probs, n_samples, replacement=True, generator=generator
            )

    return SamplerModule()
