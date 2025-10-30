"""Hybrid classical classifier mirroring the quantum helper interface."""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridClassifier(nn.Module):
    """
    A fully connected neural network with configurable depth and width.
    The architecture is deliberately simple to match the quantum circuit
    depth, but it can be extended with dropout or batch‑norm for
    robustness.

    Parameters
    ----------
    num_features: int
        Dimension of the input feature vector.
    depth: int
        Number of hidden layers.
    width: int, optional
        Width of each hidden layer. Defaults to ``num_features``.
    """

    def __init__(self, num_features: int, depth: int, width: int | None = None) -> None:
        super().__init__()
        self.num_features = num_features
        self.depth = depth
        self.width = width or num_features

        layers: List[nn.Module] = []
        in_dim = num_features
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, self.width))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.1))
            in_dim = self.width
        layers.append(nn.Linear(in_dim, 2))  # binary classification
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits for binary classification."""
        return self.net(x)

    def encode(self, data: torch.Tensor) -> torch.Tensor:
        """Identity encoding that matches the quantum ``ParameterVector``."""
        return data

    def run(self, data: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper to mimic the quantum interface."""
        encoded = self.encode(data)
        logits = self.forward(encoded)
        probs = F.softmax(logits, dim=-1)
        return probs.detach().cpu().numpy()


def build_classifier_circuit(num_features: int, depth: int) -> Tuple[HybridClassifier, List[int], List[int], List[int]]:
    """
    Construct a HybridClassifier and expose metadata identical to the QML
    ``build_classifier_circuit`` function.

    Returns
    -------
    model: HybridClassifier
        The instantiated neural network.
    encoding: List[int]
        Indices of input features that are passed through (identity).
    weight_sizes: List[int]
        Number of trainable parameters per linear layer, mirroring the quantum
        weight vector length.
    observables: List[int]
        Dummy list of observable indices; the QML implementation uses Pauli
        strings. Here we simply return the number of output logits.
    """
    model = HybridClassifier(num_features, depth)
    encoding = list(range(num_features))

    weight_sizes = [
        m.weight.numel() + m.bias.numel()
        for m in model.modules()
        if isinstance(m, nn.Linear)
    ]

    observables = [0, 1]  # two-class logits
    return model, encoding, weight_sizes, observables
