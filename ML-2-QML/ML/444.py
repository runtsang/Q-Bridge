"""Enhanced classical classifier mirroring the quantum helper interface with residual connections and dropout."""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualClassifier(nn.Module):
    """
    A feed‑forward network with optional residual connections and dropout.
    The architecture is intentionally simple yet expressive enough for binary
    classification tasks while still matching the metadata contract of the
    quantum helper.
    """

    def __init__(self, input_dim: int, hidden_dim: int, depth: int, dropout: float = 0.0):
        super().__init__()
        self.depth = depth
        self.dropout = dropout

        # First linear layer maps input to hidden space
        self.input_layer = nn.Linear(input_dim, hidden_dim)

        # Build residual blocks
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            block = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
            )
            self.blocks.append(block)

        # Final classification head
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.input_layer(x))
        for block in self.blocks:
            out = out + block(out)  # residual addition
        logits = self.head(out)
        return logits


def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
    """
    Construct a residual classifier and metadata similar to the quantum variant.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input feature vector.
    depth : int
        Number of residual blocks to stack.

    Returns
    -------
    nn.Module
        The constructed classifier.
    Iterable[int]
        Indices of the input features used for encoding.
    Iterable[int]
        Number of trainable parameters per layer (including the head).
    list[int]
        Indices of the output logits (here 0 and 1).
    """
    hidden_dim = max(8, num_features)  # ensure hidden space is at least as large as input
    classifier = ResidualClassifier(num_features, hidden_dim, depth, dropout=0.1)

    # Encoding indices: all input features
    encoding = list(range(num_features))

    # Parameter counts per linear layer
    weight_sizes = []
    for module in classifier.modules():
        if isinstance(module, nn.Linear):
            weight_sizes.append(module.weight.numel() + module.bias.numel())

    # Observables: indices of the two output logits
    observables = [0, 1]
    return classifier, encoding, weight_sizes, observables


__all__ = ["build_classifier_circuit"]
