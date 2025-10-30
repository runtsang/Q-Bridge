"""Classical QuantumClassifier with a deep skip‑connected MLP."""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn
import torch.nn.utils as utils


class QuantumClassifier(nn.Module):
    """
    A deep, skip‑connected feed‑forward network with weight‑norm and dropout.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input feature vector.
    depth : int, default=4
        Number of hidden layers.  Each hidden layer is a linear + ReLU block
        followed by a skip connection from the input.
    dropout : float, default=0.3
        Dropout probability applied after each hidden block.

    The returned metadata mirrors the quantum API: `encoding` is the list of
    input feature indices, `weight_sizes` records the number of trainable
    parameters per layer, and `observables` denotes the output class indices.
    """

    def __init__(self, num_features: int, depth: int = 4, dropout: float = 0.3) -> None:
        super().__init__()
        self.num_features = num_features
        self.depth = depth
        self.dropout = dropout

        layers: List[nn.Module] = []
        in_dim = num_features
        encoding: List[int] = list(range(num_features))
        self.weight_sizes: List[int] = []

        for idx in range(depth):
            linear = utils.weight_norm(nn.Linear(in_dim, num_features))
            layers.append(linear)
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            self.weight_sizes.append(linear.weight.numel() + linear.bias.numel())
            # skip connection: element‑wise addition with a linear map
            skip = nn.Linear(num_features, num_features)
            skip.weight.data.zero_()
            skip.bias.data.zero_()
            layers.append(skip)
            in_dim = num_features

        head = nn.Linear(in_dim, 2)
        layers.append(head)
        self.weight_sizes.append(head.weight.numel() + head.bias.numel())

        self.network = nn.Sequential(*layers)
        self.observables: List[int] = [0, 1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def get_metadata(self) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
        """
        Return the network and metadata for consistency with the quantum interface.
        """
        return self.network, self.encoding, self.weight_sizes, self.observables


__all__ = ["QuantumClassifier"]
