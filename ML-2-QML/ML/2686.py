"""Hybrid sampler‑classifier module combining classical sampler and feed‑forward classifier.

The module builds a lightweight sampler network followed by a configurable depth
classifier.  It exposes the same public API as the original `SamplerQNN` seed
while adding a second stage that refines the sampled distribution.  The
architecture is inspired by the classical `build_classifier_circuit` helper
and the quantum `SamplerQNN` interface, enabling seamless integration with
quantum‑classical pipelines.

Usage
-----
>>> from SamplerQNN__gen254 import SamplerQNNGen254
>>> model = SamplerQNNGen254(num_features=2, depth=2)
>>> logits = model(torch.randn(5, 2))
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Tuple, List


def _build_sampler() -> nn.Module:
    """Simple 2→4→2 softmax sampler."""
    return nn.Sequential(
        nn.Linear(2, 4),
        nn.Tanh(),
        nn.Linear(4, 2),
    )


def _build_classifier(in_dim: int, depth: int) -> Tuple[nn.Module, List[int]]:
    """Feed‑forward classifier mirroring the quantum design.

    Parameters
    ----------
    in_dim: int
        Number of input features to the classifier.
    depth: int
        Number of hidden layers.
    """
    layers: List[nn.Module] = []
    weight_sizes: List[int] = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, in_dim)
        layers.extend([linear, nn.ReLU()])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    return nn.Sequential(*layers), weight_sizes


class SamplerQNNGen254(nn.Module):
    """Hybrid sampler‑classifier network.

    The forward pass first samples a probability vector using the sampler
    sub‑network, then concatenates the original input with the sampled
    probabilities before passing them through the classifier.  This
    construction preserves the spirit of the original sampler while adding
    a classical refinement stage.
    """

    def __init__(self, num_features: int = 2, depth: int = 1) -> None:
        super().__init__()
        self.sampler = _build_sampler()
        # In the hybrid design the classifier sees the original features
        # plus the 2‑dimensional sampler output.
        combined_dim = num_features + 2
        self.classifier, self.classifier_weights = _build_classifier(combined_dim, depth)
        self.sampler_weights = sum(p.numel() for p in self.sampler.parameters())
        self.observables = list(range(2))  # placeholder for compatibility

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Return class logits.

        Parameters
        ----------
        x: torch.Tensor
            Input features of shape (batch, num_features).
        """
        # Sample probabilities
        probs = F.softmax(self.sampler(x), dim=-1)
        # Concatenate original features and sampled probs
        combined = torch.cat([x, probs], dim=-1)
        logits = self.classifier(combined)
        return logits

    def get_metadata(self) -> Tuple[List[int], List[int], List[int]]:
        """Return weight sizes and observables for downstream use."""
        return [self.sampler_weights], self.classifier_weights, self.observables


__all__ = ["SamplerQNNGen254"]
