"""Hybrid sampler/classifier for classical training.

This module provides a unified interface that can act as a
probability sampler or a binary classifier.  The architecture
mirrors the quantum counterpart so that parameters can be
shared or transferred between classical and quantum
implementations.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, List, Tuple

class HybridSamplerClassifier(nn.Module):
    """
    A hybrid neural network that can operate in two modes:

    * ``sampler`` – outputs a probability distribution over 2 classes.
    * ``classifier`` – a feed‑forward classifier with configurable depth.

    The class exposes the same metadata (encoding indices and weight
    sizes) as the quantum implementation, enabling seamless
    parameter transfer.
    """

    def __init__(self, num_features: int, depth: int = 1, mode: str = "sampler") -> None:
        """
        Parameters
        ----------
        num_features: int
            Dimensionality of the input feature vector.
        depth: int
            Number of hidden layers for the classifier mode.
        mode: str
            Either ``"sampler"`` or ``"classifier"``.
        """
        super().__init__()
        self.num_features = num_features
        self.depth = depth
        self.mode = mode.lower()

        if self.mode not in {"sampler", "classifier"}:
            raise ValueError(f"Unsupported mode {mode!r}")

        # Build the network
        layers: List[nn.Module] = []
        in_dim = num_features

        if self.mode == "classifier":
            for _ in range(depth):
                linear = nn.Linear(in_dim, num_features)
                layers.extend([linear, nn.ReLU()])
                in_dim = num_features
            head = nn.Linear(in_dim, 2)
            layers.append(head)
        else:  # sampler
            # Simple 2‑layer network with softmax output
            layers.append(nn.Linear(in_dim, 4))
            layers.append(nn.Tanh())
            layers.append(nn.Linear(4, 2))

        self.net = nn.Sequential(*layers)

        # Metadata
        self.encoding = list(range(num_features))
        self.weight_sizes = self._compute_weight_sizes()
        self.observables = [None] * 2  # placeholder for compatibility

    def _compute_weight_sizes(self) -> List[int]:
        sizes: List[int] = []
        for m in self.net:
            if isinstance(m, nn.Linear):
                sizes.append(m.weight.numel() + m.bias.numel())
        return sizes

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Forward pass producing logits or probabilities."""
        logits = self.net(inputs)
        if self.mode == "sampler":
            return F.softmax(logits, dim=-1)
        return logits

    # Compatibility helpers
    def get_encoding(self) -> List[int]:
        """Return the encoding indices used by the quantum counterpart."""
        return self.encoding

    def get_weight_sizes(self) -> List[int]:
        """Return a list of parameter counts per linear layer."""
        return self.weight_sizes

    def get_observables(self):
        """Return placeholder observables for compatibility."""
        return self.observables

__all__ = ["HybridSamplerClassifier"]
