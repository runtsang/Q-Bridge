"""Classical classifier factory with configurable architecture and dropout.

This class mirrors the quantum helper interface while adding support for
arbitrary hidden layer sizes, optional dropout, and batch normalization.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumClassifierModel:
    """Classical neural network classifier compatible with the quantum API."""

    @staticmethod
    def build_classifier_circuit(
        num_features: int,
        depth: int,
        hidden_sizes: List[int] | None = None,
        dropout: float = 0.0,
    ) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
        """
        Construct a feed‑forward network and return metadata.

        Parameters
        ----------
        num_features:
            Number of input features / qubits.
        depth:
            Number of linear layers to stack (excluding the head).
        hidden_sizes:
            Optional list of hidden layer widths. If None, all layers use
            ``num_features`` units.
        dropout:
            Dropout probability applied after each hidden layer.

        Returns
        -------
        network:
            ``torch.nn.Sequential`` instance.
        encoding:
            List of input indices (identity mapping).
        weight_sizes:
            Number of trainable parameters per linear layer.
        observables:
            Dummy output indices (0, 1) for the binary head.
        """
        # Decide layer widths
        if hidden_sizes is None:
            hidden_sizes = [num_features] * depth
        else:
            # Pad or truncate to match depth
            if len(hidden_sizes) < depth:
                hidden_sizes += [hidden_sizes[-1]] * (depth - len(hidden_sizes))
            hidden_sizes = hidden_sizes[:depth]

        layers: List[nn.Module] = []
        in_dim = num_features
        weight_sizes: List[int] = []

        for out_dim in hidden_sizes:
            lin = nn.Linear(in_dim, out_dim)
            layers.append(lin)
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            weight_sizes.append(lin.weight.numel() + lin.bias.numel())
            in_dim = out_dim

        # Binary classification head
        head = nn.Linear(in_dim, 2)
        layers.append(head)
        weight_sizes.append(head.weight.numel() + head.bias.numel())

        network = nn.Sequential(*layers)

        # Metadata
        encoding = list(range(num_features))
        observables = [0, 1]  # placeholder for class labels

        return network, encoding, weight_sizes, observables

__all__ = ["QuantumClassifierModel"]
