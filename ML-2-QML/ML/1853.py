"""Extended classical classifier factory with residual, dropout and layer‑norm layers."""
from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_classifier_circuit(
    num_features: int,
    depth: int,
    dropout: float = 0.1,
    use_layernorm: bool = True,
) -> Tuple[nn.Module, List[int], List[int], List[str]]:
    """
    Construct a deep residual classifier with optional dropout and layer‑norm.
    Parameters
    ----------
    num_features : int
        Number of input features.
    depth : int
        Number of residual blocks.
    dropout : float, optional
        Dropout probability applied after each block.
    use_layernorm : bool, optional
        Whether to prepend a LayerNorm before the first block.
    Returns
    -------
    network : nn.Module
        Sequential model ready for training.
    encoding : List[int]
        Indices of the input features (identity mapping).
    weight_sizes : List[int]
        Flattened weight counts for each linear layer.
    observables : List[str]
        Names of the model outputs for external inspection.
    """
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: List[int] = []

    if use_layernorm:
        layers.append(nn.LayerNorm(num_features))
        weight_sizes.append(num_features * 2)  # LayerNorm weight + bias

    for _ in range(depth):
        # Residual block: Linear -> ReLU -> Linear
        linear1 = nn.Linear(in_dim, num_features)
        linear2 = nn.Linear(num_features, num_features)
        layers.extend([linear1, nn.ReLU(), linear2, nn.ReLU()])
        weight_sizes.extend(
            [linear1.weight.numel() + linear1.bias.numel(),
             linear2.weight.numel() + linear2.bias.numel()]
        )
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        in_dim = num_features

    # Classification head
    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables = ["logits", "probabilities"]
    return network, encoding, weight_sizes, observables


__all__ = ["build_classifier_circuit"]
