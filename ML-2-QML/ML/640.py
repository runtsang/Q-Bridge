"""Enhanced classical classifier mirroring the quantum helper interface with skip connections and balanced loss."""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn

__all__ = ["build_classifier_circuit"]

def build_classifier_circuit(num_features: int,
                             depth: int,
                             *,
                             hidden_dim: int | None = None,
                             skip_connection: bool = True,
                             loss_weight: float = 1.0) -> Tuple[nn.Module,
                                                                Iterable[int],
                                                                Iterable[int],
                                                                Tuple[float, torch.Tensor]]:
    """
    Build a feed‑forward network with optional residual connections and a loss weight.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input feature vector.
    depth : int
        Number of hidden layers.
    hidden_dim : int | None, optional
        Size of the hidden layers.  If ``None`` the hidden layer size equals
        ``num_features``.
    skip_connection : bool, optional
        If ``True`` a residual connection from the input is added to each
        hidden layer output.
    loss_weight : float, optional
        Weight that is returned as part of the metadata and can be used by a
        downstream hybrid loss routine.

    Returns
    -------
    network : nn.Module
        The constructed feed‑forward network.
    encoding : Iterable[int]
        Indices of the input features that are used directly before the hidden
        layers.
    weight_sizes : Iterable[int]
        Number of trainable parameters per layer, useful for profiling.
    metadata : Tuple[float, torch.Tensor]
        A tuple containing the loss weight and a dummy tensor that keeps the
        function signature backwards compatible with the quantum version.
    """
    if hidden_dim is None:
        hidden_dim = num_features

    linear_layers = nn.ModuleList()
    weight_sizes: list[int] = []

    # Build hidden layers
    for i in range(depth):
        in_dim = num_features if i == 0 else hidden_dim
        linear = nn.Linear(in_dim, hidden_dim)
        linear_layers.append(linear)
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())

    head = nn.Linear(hidden_dim, 2)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    class ResidualClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_layers = linear_layers
            self.head = head
            self.skip_connection = skip_connection

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = x
            for linear in self.linear_layers:
                out = linear(out)
                if self.skip_connection:
                    out = out + x
                out = nn.functional.relu(out)
            out = self.head(out)
            return out

    network = ResidualClassifier()
    encoding = list(range(num_features))
    observables = (loss_weight, torch.tensor(0.0))

    return network, encoding, weight_sizes, observables
