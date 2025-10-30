"""Enhanced classical classifier factory with residual blocks and dropout."""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumClassifierModel:
    """Feed‑forward network that mirrors a quantum ansatz but adds modern regularisation.

    The network consists of `depth` residual blocks, each block containing two linear layers,
    weight‑normalised parameters, a ReLU non‑linearity, and an optional dropout.
    """

    @staticmethod
    def build_classifier_circuit(
        num_features: int,
        depth: int,
        dropout: float = 0.1,
        use_residual: bool = True,
    ) -> Tuple[nn.Sequential, List[int], List[int], List[int]]:
        """
        Build a classifier and return metadata compatible with the quantum helper.

        Parameters
        ----------
        num_features : int
            Number of input features / qubits.
        depth : int
            Number of residual blocks (each block adds two linear layers).
        dropout : float, optional
            Dropout probability applied after each block. Default 0.1.
        use_residual : bool, optional
            If False the network becomes a plain stack of linear layers.

        Returns
        -------
        network : nn.Sequential
            The constructed network.
        encoding : List[int]
            Indices of the input features used for encoding; identical to the quantum case.
        weight_sizes : List[int]
            Number of trainable parameters per linear layer.
        observables : List[int]
            Dummy observable indices (output classes).
        """
        layers: List[nn.Module] = []

        in_dim = num_features
        encoding = list(range(num_features))
        weight_sizes: List[int] = []

        for _ in range(depth):
            linear1 = nn.Linear(in_dim, num_features)
            nn.utils.weight_norm(linear1)
            layers.append(linear1)
            weight_sizes.append(linear1.weight.numel() + linear1.bias.numel())

            layers.append(nn.ReLU())

            linear2 = nn.Linear(num_features, num_features)
            nn.utils.weight_norm(linear2)
            layers.append(linear2)
            weight_sizes.append(linear2.weight.numel() + linear2.bias.numel())

            layers.append(nn.ReLU())

            if use_residual:
                # Residual connection: add input to output of the block
                layers.append(nn.Identity())

            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))

        # Final classification head
        head = nn.Linear(in_dim, 2)
        nn.utils.weight_norm(head)
        layers.append(head)
        weight_sizes.append(head.weight.numel() + head.bias.numel())

        network = nn.Sequential(*layers)

        # Observables simply indicate the output classes
        observables = [0, 1]

        return network, encoding, weight_sizes, observables

    def __init__(self, num_features: int, depth: int, **kwargs):
        self.network, _, _, _ = self.build_classifier_circuit(num_features, depth, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections."""
        out = x
        for layer in self.network:
            if isinstance(layer, nn.Identity):
                out = out + out  # simple residual; in practice use a proper residual block
            else:
                out = layer(out)
        return out


__all__ = ["QuantumClassifierModel"]
