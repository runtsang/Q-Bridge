from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Simple residual block: linear → ReLU → addition with input → dropout.
    Assumes input and output dimensions are equal.
    """
    def __init__(self, linear: nn.Linear, dropout: float):
        super().__init__()
        self.linear = linear
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.relu(self.linear(x) + x))


class ClassifierCircuitFactory:
    """
    Build a classical feed‑forward classifier that mimics the interface of the quantum circuit factory.
    """

    @staticmethod
    def build_classifier_circuit(
        num_features: int,
        depth: int,
        dropout: float = 0.0,
        residual: bool = False,
    ) -> Tuple[nn.Sequential, List[int], List[int], List[int]]:
        """
        Construct a feed‑forward network with optional residual connections and dropout.

        Parameters
        ----------
        num_features : int
            Number of input features (also the hidden width).
        depth : int
            Number of hidden layers.
        dropout : float, optional
            Dropout probability after each hidden layer.
        residual : bool, optional
            If True, each hidden block adds its input to the linear output.

        Returns
        -------
        network : nn.Sequential
            The constructed classifier.
        encoding : List[int]
            Identity mapping from input features to network input.
        weight_sizes : List[int]
            Number of trainable parameters per layer (linear + bias).
        observables : List[int]
            Indices of the output logits (binary classification → [0, 1]).
        """
        layers: List[nn.Module] = []
        weight_sizes: List[int] = []

        # Identity encoding
        encoding = list(range(num_features))

        in_dim = num_features
        for _ in range(depth):
            linear = nn.Linear(in_dim, num_features)
            weight_sizes.append(linear.weight.numel() + linear.bias.numel())

            if residual:
                layers.append(ResidualBlock(linear, dropout))
            else:
                layers.append(linear)
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))

            in_dim = num_features

        # Output head
        head = nn.Linear(in_dim, 2)
        weight_sizes.append(head.weight.numel() + head.bias.numel())
        layers.append(head)

        network = nn.Sequential(*layers)
        observables = [0, 1]
        return network, encoding, weight_sizes, observables


__all__ = ["ClassifierCircuitFactory"]
