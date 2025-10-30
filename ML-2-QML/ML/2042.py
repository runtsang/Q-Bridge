"""Hybrid classical classifier with dropout and advanced weight‑sharing.

The ML component is expanded from a simple feed‑forward stack to a
fully‑connected network that supports weight sharing across multiple
sub‑blocks.  Dropout is applied after every ReLU, and the network is
exposed as a ``Classifier`` class that can be instantiated and
trained with any PyTorch ``DataLoader``.  The class also exposes a
``forward`` method that returns the logits and the intermediate
activations for use in a hybrid training loop.
"""

from __future__ import annotations

import math
from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    """Class‑based interface for a multi‑layer feed‑forward network.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input features.
    depth : int
        Number of hidden layers.
    hidden_dim : int, optional
        Width of each hidden layer (default: same as ``num_features``).
    dropout : float, optional
        Dropout probability applied after each ReLU (default: 0.0).
    share_weights : bool, optional
        If True, all hidden layers share the same weight matrix.
    """
    def __init__(
        self,
        num_features: int,
        depth: int,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
        share_weights: bool = False,
    ):
        super().__init__()
        hidden_dim = hidden_dim or num_features
        self.num_features = num_features
        self.depth = depth
        self.dropout = dropout
        self.share_weights = share_weights

        # Build the hidden layers
        if share_weights:
            shared_linear = nn.Linear(num_features, hidden_dim, bias=True)
            self.hidden_layers = nn.ModuleList(
                [shared_linear for _ in range(depth)]
            )
        else:
            self.hidden_layers = nn.ModuleList(
                [nn.Linear(num_features if i == 0 else hidden_dim, hidden_dim) for i in range(depth)]
            )

        self.activation = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass returning logits and hidden activations.

        Parameters
        ----------
        x
            Input tensor of shape ``(batch, num_features)``.

        Returns
        -------
        logits
            Logits of shape ``(batch, 2)``.
        activations
            List of hidden activations, one per layer, for use in a hybrid
            training loop.
        """
        activations: list[torch.Tensor] = []
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.activation(x)
            x = self.dropout_layer(x)
            activations.append(x)
        logits = self.head(x)
        return logits, activations

def build_classifier_circuit(
    num_features: int,
    depth: int,
    dropout: float = 0.0,
    share_weights: bool = False,
) -> Tuple[Classifier, Iterable[int], Iterable[int], list[int]]:
    """Construct a feed‑forward network with optional dropout and shared weights.

    Returns the same tuple shape as the original seed: model, encoding,
    weight_sizes, observables.  ``encoding`` and ``observables`` are
    placeholders that can be consumed by a quantum module.
    """
    model = Classifier(
        num_features=num_features,
        depth=depth,
        dropout=dropout,
        share_weights=share_weights,
    )
    # In the classical network we expose the indices of the trainable
    # parameters in the same way as the quantum variant.  ``encoding``
    # holds the indices of the input weights; ``weight_sizes`` holds
    # the number of trainable parameters per layer; ``observables`` is
    # a dummy list of size 2 (for a 2‑class problem).
    encoding = list(range(num_features))
    weight_sizes = []
    for layer in model.hidden_layers:
        weight_sizes.append(layer.weight.numel() + layer.bias.numel())
    weight_sizes.append(model.head.weight.numel() + model.head.bias.numel())
    observables = list(range(2))
    return model, encoding, weight_sizes, observables

__all__ = ["build_classifier_circuit", "Classifier"]
