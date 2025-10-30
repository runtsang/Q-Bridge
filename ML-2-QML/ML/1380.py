"""Extended classical classifier factory with residual connections and optional dropout."""

from __future__ import annotations

from typing import Iterable, Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """A simple residual block: Linear → ReLU → Linear + skip."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        self.skip = nn.Linear(in_features, out_features) if in_features!= out_features else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = F.relu(self.linear1(x))
        out = self.linear2(out)
        return F.relu(out + residual)


def build_classifier_circuit(
    num_features: int,
    depth: int,
    num_classes: int = 2,
    dropout: Optional[float] = None,
    activation: str = "relu",
) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """
    Construct a deep residual classifier.

    Parameters
    ----------
    num_features : int
        Number of input features.
    depth : int
        Number of residual blocks.
    num_classes : int, default 2
        Number of output classes.
    dropout : float | None, default None
        Dropout probability applied after each block.
    activation : str, default "relu"
        Activation function to use; currently only "relu" is supported.

    Returns
    -------
    network : nn.Module
        The constructed residual network.
    encoding : Iterable[int]
        Feature indices used for encoding (identity mapping).
    weight_sizes : Iterable[int]
        Number of trainable parameters per linear layer.
    observables : List[int]
        Indices of output neurons corresponding to class logits.
    """
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: List[int] = []

    for _ in range(depth):
        block = ResidualBlock(in_dim, num_features)
        layers.append(block)
        weight_sizes.extend([p.numel() for p in block.parameters()])
        if dropout is not None:
            layers.append(nn.Dropout(dropout))

    head = nn.Linear(num_features, num_classes)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables = list(range(num_classes))
    return network, encoding, weight_sizes, observables


__all__ = ["build_classifier_circuit", "ResidualBlock"]
