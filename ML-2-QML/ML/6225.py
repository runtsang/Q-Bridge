"""Enhanced classical classifier that mirrors the quantum helper and exposes training metadata."""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumClassifierModel(nn.Module):
    """Feed‑forward classifier that mimics the structure of the quantum helper.

    The network consists of ``depth`` identical blocks of ``Linear → ReLU`` followed by a
    final linear layer to two outputs.  The method ``metadata`` returns the same
    encoding, weight‑size list and observable indices that the quantum routine
    produces, enabling side‑by‑side comparison in experiments.

    Parameters
    ----------
    num_features : int
        Size of the input feature vector.
    depth : int
        Number of hidden layers.
    hidden_size : int, optional
        Width of each hidden layer; defaults to ``num_features`` for symmetry.
    """

    def __init__(self, num_features: int, depth: int, hidden_size: int | None = None) -> None:
        super().__init__()
        hidden = hidden_size or num_features
        layers: List[nn.Module] = []
        in_dim = num_features
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def metadata(self) -> Tuple[Iterable[int], Iterable[int], List[int]]:
        """Return encoding indices, weight‑size list and observable indices.

        The encoding is simply the identity mapping ``range(num_features)``.
        Weight sizes are computed from each linear layer.  Observables are the two
        output logits, so ``range(2)``.
        """
        encoding = list(range(self.network[0].in_features))
        weight_sizes: List[int] = []
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                weight_sizes.append(layer.weight.numel() + layer.bias.numel())
        observables = list(range(2))
        return encoding, weight_sizes, observables


def SamplerQNN() -> nn.Module:
    """Small neural sampler that mirrors the quantum SamplerQNN helper."""
    class _Sampler(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 4),
                nn.Tanh(),
                nn.Linear(4, 2),
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return F.softmax(self.net(inputs), dim=-1)

    return _Sampler()


__all__ = ["QuantumClassifierModel", "SamplerQNN"]
