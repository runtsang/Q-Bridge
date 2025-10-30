"""Enhanced classical classifier with multi‑head support, dropout, and L1 regularisation."""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn

class MultiHeadClassifier(nn.Module):
    """A feed‑forward network with a shared backbone and multiple classification heads."""
    def __init__(
        self,
        num_features: int,
        depth: int,
        num_heads: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = num_features
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, num_features))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = num_features
        self.backbone = nn.Sequential(*layers)
        self.heads = nn.ModuleList([nn.Linear(num_features, 2) for _ in range(num_heads)])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor,...]:
        h = self.backbone(x)
        return tuple(head(h) for head in self.heads)

def build_classifier_circuit(
    num_features: int,
    depth: int,
    *,
    num_heads: int = 1,
    dropout: float = 0.0,
    l1_reg: float = 0.0,
) -> Tuple[
    nn.Module,
    Iterable[int],
    Iterable[int],
    List[int],
]:
    """
    Construct a feed‑forward network that mirrors the quantum circuit builder.
    The ``num_heads`` parameter allows for multi‑head classification.
    ``dropout`` adds regularisation between layers.
    ``l1_reg`` is stored for external use but not applied inside the network.
    """
    network = MultiHeadClassifier(
        num_features=num_features,
        depth=depth,
        num_heads=num_heads,
        dropout=dropout,
    )
    encoding = list(range(num_features))
    weight_sizes: List[int] = []
    for module in network.modules():
        if isinstance(module, nn.Linear):
            weight_sizes.append(module.weight.numel() + module.bias.numel())
    observables = list(range(num_heads))
    return network, encoding, weight_sizes, observables
