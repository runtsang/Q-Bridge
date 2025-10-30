"""Classical neural network classifier mirroring the quantum interface.

The model is a simple feed‑forward network that emulates the structure
returned by the quantum build_classifier_circuit helper.  It exposes
the same API (`encoding`, `weight_sizes`, `observables`) so that
experiments can interchange the classical and quantum back‑ends
without changing the surrounding code.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn


def build_classifier_circuit(num_features: int, depth: int,
                             hidden_dim: int = 32) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """
    Build a classical feed‑forward classifier that mimics the quantum
    circuit metadata: *encoding*, *weight_sizes* and *observables*.

    Parameters
    ----------
    num_features:
        Dimensionality of the input feature vector.
    depth:
        Number of hidden layers.
    hidden_dim:
        Width of each hidden layer.

    Returns
    -------
    network:
        nn.Sequential model.
    encoding:
        Identity mapping of input indices.
    weight_sizes:
        Number of trainable parameters per linear layer.
    observables:
        Output class indices.
    """
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: List[int] = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, hidden_dim)
        layers.extend([linear, nn.ReLU()])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = hidden_dim

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables = [0, 1]  # two‑class classification
    return network, encoding, weight_sizes, observables


class QuantumClassifierModel(nn.Module):
    """
    Classical counterpart of the quantum classifier.

    The class implements the same public attributes as the quantum
    implementation (encoding, weight_sizes, observables) so that
    higher‑level code can swap the back‑end seamlessly.

    Parameters
    ----------
    num_features:
        Input dimensionality.
    depth:
        Number of hidden layers.
    hidden_dim:
        Width of hidden layers.
    """

    def __init__(self, num_features: int, depth: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net, self.encoding, self.weight_sizes, self.observables = build_classifier_circuit(
            num_features, depth, hidden_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def parameters(self):
        return super().parameters()


__all__ = ["QuantumClassifierModel", "build_classifier_circuit"]
