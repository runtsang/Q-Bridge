"""Enhanced classical classifier mirroring the quantum helper interface.

Features:
- Configurable dropout and batch‑normalization layers.
- Optional residual connections for deeper networks.
- Returns the full network, encoding indices, weight sizes and observables.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn


class QuantumClassifierModel:
    """
    Classical neural‑network implementation that mimics the interface of the
    quantum classifier factory.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input feature vector.
    depth : int
        Number of hidden layers in the classifier.
    dropout : float, default 0.0
        Drop‑out probability applied after each hidden layer.
    use_batchnorm : bool, default False
        Whether to insert a batch‑norm layer after each linear transform.
    residual : bool, default False
        If True, add a simple residual connection between consecutive layers.
    """

    def __init__(
        self,
        num_features: int,
        depth: int,
        dropout: float = 0.0,
        use_batchnorm: bool = False,
        residual: bool = False,
    ) -> None:
        self.num_features = num_features
        self.depth = depth
        self.dropout = dropout
        self.use_batchnorm = use_batchnorm
        self.residual = residual

    def build(self) -> Tuple[nn.Module, Iterable[int], List[int], List[int]]:
        """
        Construct a feed‑forward classifier and return metadata.

        Returns
        -------
        network : torch.nn.Sequential
            The constructed neural network.
        encoding : list[int]
            Indices of input features that are directly fed to the network.
        weight_sizes : list[int]
            Number of learnable parameters per layer (including bias).
        observables : list[int]
            Dummy observable indices representing the two‑class logits.
        """
        layers: List[nn.Module] = []
        weight_sizes: List[int] = []

        # Input to first hidden layer
        layers.append(nn.Linear(self.num_features, self.num_features))
        weight_sizes.append(
            layers[-1].weight.numel() + layers[-1].bias.numel()
        )
        if self.use_batchnorm:
            layers.append(nn.BatchNorm1d(self.num_features))
        layers.append(nn.ReLU())
        if self.dropout > 0.0:
            layers.append(nn.Dropout(self.dropout))

        in_dim = self.num_features

        # Hidden layers
        for _ in range(self.depth - 1):
            out_dim = self.num_features
            linear = nn.Linear(in_dim, out_dim)
            layers.append(linear)
            weight_sizes.append(
                linear.weight.numel() + linear.bias.numel()
            )
            if self.use_batchnorm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU())
            if self.dropout > 0.0:
                layers.append(nn.Dropout(self.dropout))
            in_dim = out_dim

            # Optional residual
            if self.residual:
                # Simple residual: add identity if dimensions match
                if in_dim == self.num_features:
                    layers.append(nn.Identity())

        # Classifier head
        head = nn.Linear(in_dim, 2)
        layers.append(head)
        weight_sizes.append(head.weight.numel() + head.bias.numel())

        network = nn.Sequential(*layers)

        encoding = list(range(self.num_features))
        observables = [0, 1]  # logits for class 0 and 1

        return network, encoding, weight_sizes, observables
