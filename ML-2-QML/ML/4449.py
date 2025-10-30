"""Hybrid quantum‑graph classifier – classical implementation."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import networkx as nx
from torch import Tensor

# Local utilities – identical signatures to the original seeds
from GraphQNN import (
    feedforward as _feedforward,
    fidelity_adjacency as _fidelity_adjacency,
    random_network as _random_network,
    random_training_data as _random_training_data,
    state_fidelity as _state_fidelity,
)
from FastBaseEstimator import FastBaseEstimator


def build_classifier_circuit(
    num_features: int, depth: int
) -> Tuple[nn.Module, List[int], List[int], List[int]]:
    """
    Construct a simple feed‑forward classifier and expose metadata
    comparable to the quantum counterpart.

    Parameters
    ----------
    num_features : int
        Size of the input layer.
    depth : int
        Number of hidden layers.

    Returns
    -------
    network : nn.Module
        Sequential network.
    encoding : List[int]
        Indices of input features used for encoding.
    weight_sizes : List[int]
        Number of trainable parameters per layer.
    observables : List[int]
        Dummy observable indices (classical version).
    """
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: List[int] = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.extend([linear, nn.ReLU()])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables = list(range(2))
    return network, encoding, weight_sizes, observables


class HybridQuantumGraphClassifier:
    """
    Classical side of the hybrid classifier.
    Builds a neural net, propagates hidden states, and constructs a fidelity‑based graph.
    """

    def __init__(self, arch: Sequence[int]) -> None:
        """
        Parameters
        ----------
        arch
            Layer sizes, e.g. ``[num_features, 16, 8, 2]``.
        """
        self.arch = list(arch)
        self.net = self._build_network()
        self.estimator = FastBaseEstimator(self.net)

    def _build_network(self) -> nn.Module:
        layers: List[nn.Module] = []
        in_dim = self.arch[0]
        for out_dim in self.arch[1:-1]:
            layers.extend([nn.Linear(in_dim, out_dim), nn.Tanh()])
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, self.arch[-1]))
        return nn.Sequential(*layers)

    def random_training_data(self, samples: int = 200) -> List[Tuple[Tensor, Tensor]]:
        """Generate synthetic data using the target layer weights."""
        target_weight = self.net[-1].weight
        return _random_training_data(target_weight, samples)

    def train(self, data: List[Tuple[Tensor, Tensor]], lr: float = 0.01, epochs: int = 10) -> None:
        """Simple SGD training loop."""
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        self.net.train()
        for _ in range(epochs):
            for x, y in data:
                optimizer.zero_grad()
                pred = self.net(x)
                loss = loss_fn(pred, y)
                loss.backward()
                optimizer.step()

    def graph_of_hidden_states(self, data: Iterable[Tuple[Tensor, Tensor]]) -> nx.Graph:
        """Return a graph where nodes are hidden states and edges are weighted by fidelity."""
        states = _feedforward(self.arch, [layer.weight for layer in self.net], data)
        # Flatten to a list of tensors per layer (exclude input)
        flat_states = [s for layer in states for s in layer[1:]]
        return _fidelity_adjacency(flat_states, threshold=0.9)

    def evaluate(self, inputs: Iterable[Tensor]) -> List[float]:
        """Return predictions for a batch of inputs."""
        preds: List[float] = []
        self.net.eval()
        with torch.no_grad():
            for x in inputs:
                preds.append(float(self.net(x).item()))
        return preds

    def build_classifier_circuit(self, depth: int = 2) -> Tuple[nn.Module, List[int], List[int], List[int]]:
        """Return network, encoding indices, weight sizes, and output observables."""
        return build_classifier_circuit(num_features=self.arch[0], depth=depth)


__all__ = ["HybridQuantumGraphClassifier", "build_classifier_circuit"]
