"""
Classical classifier factory mirroring the quantum helper interface with additional regularisation options.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List, Optional

import torch
import torch.nn as nn


class QuantumClassifierModel:
    """
    Factory for a feed‑forward neural network that mimics the structure of the quantum ansatz.

    Parameters
    ----------
    num_features : int
        Number of input features / qubits.
    depth : int
        Number of hidden layers in the network.
    dropout : Optional[float]
        Dropout probability. If ``None`` no dropout is applied.
    batchnorm : bool
        Whether to insert a BatchNorm1d after each hidden layer.
    """

    @staticmethod
    def build_classifier_circuit(
        num_features: int,
        depth: int,
        dropout: Optional[float] = None,
        batchnorm: bool = False,
    ) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
        """
        Construct a feed‑forward classifier and metadata similar to the quantum variant.

        Returns
        -------
        network : nn.Sequential
            The constructed model.
        encoding : Iterable[int]
            Indices of features that are linearly mapped to the first hidden layer.
        weight_sizes : Iterable[int]
            Number of trainable parameters per layer (including bias).
        observables : List[int]
            Dummy observable indices for compatibility with the quantum interface.
        """
        layers: List[nn.Module] = []
        in_dim = num_features
        encoding = list(range(num_features))
        weight_sizes: List[int] = []

        for _ in range(depth):
            linear = nn.Linear(in_dim, num_features)
            layers.append(linear)
            weight_sizes.append(linear.weight.numel() + linear.bias.numel())

            if batchnorm:
                layers.append(nn.BatchNorm1d(num_features))

            layers.append(nn.ReLU())

            if dropout is not None:
                layers.append(nn.Dropout(p=dropout))

            in_dim = num_features

        head = nn.Linear(in_dim, 2)
        layers.append(head)
        weight_sizes.append(head.weight.numel() + head.bias.numel())

        network = nn.Sequential(*layers)

        # Observables are placeholder indices that the quantum side expects.
        observables = list(range(2))
        return network, encoding, weight_sizes, observables


__all__ = ["QuantumClassifierModel"]
