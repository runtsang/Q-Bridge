"""QuantumClassifierModel: Classical feed‑forward classifier with metadata.

The class mirrors the quantum helper interface so that a hybrid training loop can
pass the same descriptor objects (encoding, weight sizes, observables) to both
the classical and quantum components.  The network is built with a configurable
depth and feature dimension and exposes helper properties that expose the
underlying parameter counts, mimicking the quantum implementation.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn


class QuantumClassifierModel(nn.Module):
    """Feed‑forward classifier with depth‑controlled hidden layers.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input feature vector.
    depth : int
        Number of hidden layers.  Each hidden layer maps to ``num_features`` units.
    """

    def __init__(self, num_features: int, depth: int) -> None:
        super().__init__()
        self.num_features = num_features
        self.depth = depth

        layers: List[nn.Module] = []
        in_dim = num_features
        for _ in range(depth):
            linear = nn.Linear(in_dim, num_features)
            layers.append(linear)
            layers.append(nn.ReLU())
            in_dim = num_features

        # Output head
        layers.append(nn.Linear(in_dim, 2))
        self.net = nn.Sequential(*layers)

        # Metadata
        self.encoding = list(range(num_features))
        self.weight_sizes = self._compute_weight_sizes()
        self.observables = list(range(2))

    def _compute_weight_sizes(self) -> List[int]:
        """Return the number of trainable parameters per layer."""
        sizes: List[int] = []
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                sizes.append(layer.weight.numel() + layer.bias.numel())
        return sizes

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)

    @property
    def encoding_meta(self) -> Iterable[int]:
        """Return the feature indices used for encoding."""
        return self.encoding

    @property
    def weight_meta(self) -> Iterable[int]:
        """Return the parameter counts per linear layer."""
        return self.weight_sizes

    @property
    def observable_meta(self) -> Iterable[int]:
        """Return a placeholder observable list."""
        return self.observables


def build_classifier_circuit(num_features: int, depth: int) -> Tuple[QuantumClassifierModel, Iterable[int], Iterable[int], List[int]]:
    """Factory that returns a ``QuantumClassifierModel`` instance and its metadata.

    The returned tuple matches the signature of the quantum helper, enabling
    a unified interface for hybrid pipelines.

    Parameters
    ----------
    num_features : int
        Input dimensionality.
    depth : int
        Number of hidden layers.

    Returns
    -------
    model : QuantumClassifierModel
        The instantiated feed‑forward network.
    encoding : Iterable[int]
        Feature indices used for encoding.
    weight_sizes : Iterable[int]
        Number of trainable parameters per linear layer.
    observables : List[int]
        Placeholder observable identifiers.
    """
    model = QuantumClassifierModel(num_features, depth)
    return model, model.encoding_meta, model.weight_meta, model.observable_meta


__all__ = ["QuantumClassifierModel", "build_classifier_circuit"]
