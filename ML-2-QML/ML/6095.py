"""Extended classical classifier with dropout, batch‑norm, and residual connections.

The `QuantumClassifierModel` class mirrors the quantum helper interface while adding richer
architectural components. It exposes a static `build_classifier_circuit` method that
returns a PyTorch `nn.Sequential` network, a list of feature indices used for encoding,
the sizes of all trainable weight tensors, and a list of final output labels.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn


class QuantumClassifierModel:
    """Classical feed‑forward classifier with dropout and batch‑norm support."""

    @staticmethod
    def build_classifier_circuit(
        num_features: int,
        depth: int,
        dropout: float = 0.5,
        use_batchnorm: bool = True,
    ) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
        """
        Construct a deep feed‑forward network.

        Parameters
        ----------
        num_features : int
            Number of input features / qubits.
        depth : int
            Number of hidden blocks.
        dropout : float, optional
            Dropout probability applied after each hidden block.
        use_batchnorm : bool, optional
            Whether to add a BatchNorm1d layer after each linear layer.

        Returns
        -------
        network : nn.Module
            The constructed network.
        encoding : Iterable[int]
            Indices of input features used for the encoding stage.
        weight_sizes : Iterable[int]
            Number of trainable parameters per linear layer.
        observables : List[int]
            Dummy observable indices; mirrors the quantum interface.
        """
        layers: List[nn.Module] = []
        in_dim = num_features
        weight_sizes: List[int] = []

        # encoding layer simply forwards the input
        encoding = list(range(num_features))

        for _ in range(depth):
            linear = nn.Linear(in_dim, num_features, bias=True)
            layers.append(linear)
            weight_sizes.append(linear.weight.numel() + linear.bias.numel())

            if use_batchnorm:
                layers.append(nn.BatchNorm1d(num_features))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

            in_dim = num_features

        # final classification head
        head = nn.Linear(in_dim, 2, bias=True)
        layers.append(head)
        weight_sizes.append(head.weight.numel() + head.bias.numel())

        network = nn.Sequential(*layers)
        observables = [0, 1]  # placeholder for classical analog of quantum observables
        return network, encoding, weight_sizes, observables


__all__ = ["QuantumClassifierModel"]
