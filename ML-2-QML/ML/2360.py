"""Hybrid classical estimator that mirrors the quantum estimator interface.

The class exposes a feed‑forward network with configurable depth and
output dimension.  It also returns metadata (encoding indices,
weight sizes, observables) that can be consumed by the quantum
counterpart.  The design is inspired by the original EstimatorQNN
regressor and the classifier factory from QuantumClassifierModel.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Iterable, Tuple, List

def build_classical_circuit(num_features: int,
                            depth: int,
                            output_dim: int = 1) -> Tuple[nn.Module,
                                                          Iterable[int],
                                                          Iterable[int],
                                                          List[int]]:
    """
    Construct a fully‑connected network that mirrors the quantum
    circuit builder.

    Parameters
    ----------
    num_features: int
        Number of input features.
    depth: int
        Number of hidden layers.
    output_dim: int, default 1
        Dimensionality of the output (1 for regression, 2 for binary
        classification).

    Returns
    -------
    network: nn.Module
        The stacked network.
    encoding: Iterable[int]
        Indices of input features that are treated as encoding
        parameters in the quantum module.
    weight_sizes: Iterable[int]
        Number of trainable parameters per linear layer.
    observables: List[int]
        Dummy observable indices that match the quantum observables
        list; useful when the quantum module expects a list of
        measurement operators.
    """
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: List[int] = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.append(linear)
        layers.append(nn.Tanh())
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features

    head = nn.Linear(in_dim, output_dim)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables = list(range(output_dim))
    return network, encoding, weight_sizes, observables

class EstimatorQNNHybrid(nn.Module):
    """
    Classical estimator that mimics the API of the quantum EstimatorQNN.

    The network can be used for regression or binary classification
    depending on the ``output_dim`` passed at construction time.
    """

    def __init__(self,
                 num_features: int,
                 depth: int = 2,
                 output_dim: int = 1) -> None:
        super().__init__()
        self.net, self.encoding, self.weight_sizes, self.observables = build_classical_circuit(
            num_features, depth, output_dim
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass through the stacked network.

        Parameters
        ----------
        inputs: torch.Tensor
            Input tensor of shape ``(batch, num_features)``.
        """
        return self.net(inputs)

    @property
    def metadata(self) -> Tuple[Iterable[int], Iterable[int], List[int]]:
        """
        Return the encoding indices, weight sizes and observables.
        """
        return self.encoding, self.weight_sizes, self.observables

__all__ = ["EstimatorQNNHybrid", "build_classical_circuit"]
