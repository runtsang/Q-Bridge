"""Classical classifier factory with extended capabilities.

The class provides a feed‑forward network builder that supports optional dropout
and layer‑normalisation, mimicking the interface of the quantum helper while
offering richer classical variants.  The returned metadata can be used for
parameter counting, model inspection or auto‑generation of training scripts.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn


class QuantumClassifierModel:
    """
    Factory for classical feed‑forward classification networks.
    """

    @staticmethod
    def build_classifier_circuit(
        num_features: int,
        depth: int,
        *,
        dropout: float = 0.0,
        use_layernorm: bool = False,
    ) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[torch.Tensor]]:
        """
        Construct a multi‑layer perceptron with optional dropout and layer‑norm.

        Parameters
        ----------
        num_features : int
            Dimensionality of the input feature vector.
        depth : int
            Number of hidden layers.
        dropout : float, optional
            Dropout probability applied after each activation.  Defaults to 0.0
            (no dropout).
        use_layernorm : bool, optional
            If ``True`` a ``LayerNorm`` is inserted after each hidden linear
            layer.  Defaults to ``False``.

        Returns
        -------
        model : torch.nn.Module
            The assembled network.
        encoding : Iterable[int]
            Indices of the input features that are fed into the network.
        weight_sizes : Iterable[int]
            Linear‑layer parameter counts (weights + bias) for each layer.
        observables : List[torch.Tensor]
            Dummy observables that mirror the quantum API; useful when
            combining classical and quantum training loops.
        """
        layers: List[nn.Module] = []
        in_dim = num_features
        encoding = list(range(num_features))
        weight_sizes: List[int] = []

        for _ in range(depth):
            linear = nn.Linear(in_dim, num_features)
            layers.append(linear)
            weight_sizes.append(linear.weight.numel() + linear.bias.numel())
            if use_layernorm:
                layers.append(nn.LayerNorm(num_features))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = num_features

        # Output head
        head = nn.Linear(in_dim, 2)
        layers.append(head)
        weight_sizes.append(head.weight.numel() + head.bias.numel())

        model = nn.Sequential(*layers)

        # Dummy observables: identity matrices for each output class
        observables = [torch.eye(2) for _ in range(2)]

        return model, encoding, weight_sizes, observables


__all__ = ["QuantumClassifierModel"]
