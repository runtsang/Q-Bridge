"""Enhanced classical classifier mirroring the quantum helper interface.

The module defines `QuantumClassifierModel`, a PyTorch neural network that
supports configurable hidden layers, dropout, and optional residual
connections.  It also provides a static factory method
`build_classifier_circuit` which returns the network together with
encoding indices, weight statistics and observable labels – a
drop‑in replacement for the original seed function.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn


class QuantumClassifierModel(nn.Module):
    """
    Classical feed‑forward network with optional residual connections.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input feature vector.
    hidden_dims : Iterable[int], default (64, 32)
        Size of each hidden layer.  The number of layers is ``len(hidden_dims)``.
    dropout : float, default 0.0
        Dropout probability applied after each ReLU.  Set to 0.0 to disable.
    residual : bool, default False
        If ``True`` a skip connection is added after each linear layer when
        the input and output dimensions match.
    """

    def __init__(
        self,
        num_features: int,
        hidden_dims: Iterable[int] = (64, 32),
        dropout: float = 0.0,
        residual: bool = False,
    ) -> None:
        super().__init__()
        self.residual = residual
        layers: List[nn.Module] = []
        in_dim = num_features

        for out_dim in hidden_dims:
            linear = nn.Linear(in_dim, out_dim)
            layers.append(linear)
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = out_dim

        self.hidden = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.hidden(x)
        out = self.head(out)
        return out

    @staticmethod
    def build_classifier_circuit(
        num_features: int,
        hidden_dims: Iterable[int] = (64, 32),
        dropout: float = 0.0,
        residual: bool = False,
    ) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
        """
        Construct a network mirroring the quantum helper interface.

        Returns
        -------
        network
            :class:`torch.nn.Sequential` containing the hidden layers and head.
        encoding
            List of feature indices (identical to ``range(num_features)``).
        weight_sizes
            Number of trainable parameters per layer.
        observables
            Class indices ``[0, 1]`` representing the two output classes.
        """
        layers: List[nn.Module] = []
        in_dim = num_features
        encoding = list(range(num_features))
        weight_sizes: List[int] = []

        for out_dim in hidden_dims:
            linear = nn.Linear(in_dim, out_dim)
            layers.append(linear)
            weight_sizes.append(linear.weight.numel() + linear.bias.numel())
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = out_dim

        head = nn.Linear(in_dim, 2)
        layers.append(head)
        weight_sizes.append(head.weight.numel() + head.bias.numel())

        network = nn.Sequential(*layers)
        observables = [0, 1]
        return network, encoding, weight_sizes, observables


__all__ = ["QuantumClassifierModel"]
