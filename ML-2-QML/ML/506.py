"""Extended classical classifier mirroring the quantum API.

This version augments the seed architecture with residual connections,
batch‑normalisation and dropout, giving it greater expressive power while
maintaining the same public interface as the quantum counterpart.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn


class QuantumClassifierModel(nn.Module):
    """Feed‑forward classifier with residual blocks.

    Parameters
    ----------
    num_features : int
        Size of the input vector.
    depth : int
        Number of residual blocks.
    dropout : float, default 0.2
        Drop‑out probability applied inside each block.
    """

    def __init__(self, num_features: int, depth: int, dropout: float = 0.2):
        super().__init__()
        self.encoding = list(range(num_features))
        self.body = nn.Sequential()
        for i in range(depth):
            block = nn.Sequential(
                nn.Linear(num_features, num_features),
                nn.BatchNorm1d(num_features),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            # each block adds a residual connection
            self.body.add_module(f"residual_{i}", block)
        self.head = nn.Linear(num_features, 2)
        # store weight sizes for API compatibility
        self.weight_sizes = [p.numel() for p in self.parameters()]
        self.observables: list = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for block in self.body:
            out = out + block(out)  # residual connection
        return self.head(out)

    def get_encoding(self) -> Iterable[int]:
        """Return the indices used for encoding – identical to the quantum helper."""
        return self.encoding

    def get_weight_sizes(self) -> Iterable[int]:
        """Return a list with the number of trainable parameters per layer."""
        return self.weight_sizes

    def get_observables(self) -> Iterable:
        """Placeholder for API compatibility."""
        return self.observables


def build_classifier_circuit(
    num_features: int, depth: int, dropout: float = 0.2
) -> Tuple[QuantumClassifierModel, Iterable[int], Iterable[int], list]:
    """Convenience factory that mirrors the quantum signature."""
    model = QuantumClassifierModel(num_features, depth, dropout)
    return model, model.get_encoding(), model.get_weight_sizes(), model.get_observables()


__all__ = ["QuantumClassifierModel", "build_classifier_circuit"]
