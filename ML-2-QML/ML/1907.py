"""Graph neural network utilities with advanced fidelity graph analysis.

This module defines a :class:`GraphQNN` class that wraps a simple
tanh‑activated feed‑forward network and exposes a set of tools for
building training data, running inference, and constructing a graph
from state fidelities—exactly the functionality of the seed
implementation, but packaged as an object with convenient methods."""
from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import numpy as np

Tensor = torch.Tensor


class GraphQNN(nn.Module):
    """A simple feed‑forward neural network with tanh activations.

    The class mirrors the original procedural API while adding
    convenient methods for training, evaluation, and fidelity‑based
    graph construction.
    """

    def __init__(self, architecture: Sequence[int]) -> None:
        super().__init__()
        self.architecture = list(architecture)
        self.layers = nn.ModuleList()
        for in_f, out_f in zip(self.architecture[:-1], self.architecture[1:]):
            self.layers.append(nn.Linear(in_f, out_f))
        self.activation = nn.Tanh()

    # ------------------------------------------------------------------ #
    #  Static helpers for data generation
    # ------------------------------------------------------------------ #
    @staticmethod
    def _random_linear(in_features: int, out_features: int) -> Tensor:
        return torch.randn(out_features, in_features, dtype=torch.float32)

    @staticmethod
    def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
        dataset: List[Tuple[Tensor, Tensor]] = []
        for _ in range(samples):
            features = torch.randn(weight.size(1), dtype=torch.float32)
            target = weight @ features
            dataset.append((features, target))
        return dataset

    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int):
        """Generate a random network and a matching training set."""
        weights: List[Tensor] = []
        for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
            weights.append(GraphQNN._random_linear(in_f, out_f))
        target_weight = weights[-1]
        training_data = GraphQNN.random_training_data(target_weight, samples)
        return list(qnn_arch), weights, training_data, target_weight

    # ------------------------------------------------------------------ #
    #  Forward propagation
    # ------------------------------------------------------------------ #
    def feedforward(self, samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
        """Return activations for every layer for each sample."""
        stored: List[List[Tensor]] = []
        for features, _ in samples:
            activations = [features]
            current = features
            for layer in self.layers:
                current = self.activation(layer(current))
                activations.append(current)
            stored.append(activations)
        return stored

    def get_hidden_states(self, samples: Iterable[Tuple[Tensor, Tensor]]) -> List[Tuple[Tensor, Tensor]]:
        """Return a list of (hidden_state, target) tuples."""
        hidden_states: List[Tuple[Tensor, Tensor]] = []
        for features, target in samples:
            current = features
            for layer in self.layers:
                current = self.activation(layer(current))
            hidden_states.append((current, target))
        return hidden_states

    # ------------------------------------------------------------------ #
    #  Fidelity utilities
    # ------------------------------------------------------------------ #
    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float(torch.dot(a_norm, b_norm).item() ** 2)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNN.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    # ------------------------------------------------------------------ #
    #  Training utilities
    # ------------------------------------------------------------------ #
    def train(
        self,
        training_data: Iterable[Tuple[Tensor, Tensor]],
        epochs: int = 200,
        lr: float = 1e-3,
        loss_fn: nn.Module = nn.MSELoss(),
    ) -> None:
        optimizer = optim.Adam(self.parameters(), lr=lr)
        for _ in range(epochs):
            for features, target in training_data:
                optimizer.zero_grad()
                out = self.forward(features)
                loss = loss_fn(out, target)
                loss.backward()
                optimizer.step()

    def evaluate(self, samples: Iterable[Tuple[Tensor, Tensor]]) -> float:
        """Return mean squared error over the given samples."""
        total = 0.0
        n = 0
        with torch.no_grad():
            for features, target in samples:
                out = self.forward(features)
                total += ((out - target) ** 2).sum().item()
                n += 1
        return total / n

    # ------------------------------------------------------------------ #
    #  Utility to export weights for quantum mapping
    # ------------------------------------------------------------------ #
    def to_weight_list(self) -> List[Tensor]:
        """Return a list of weight matrices for each layer."""
        return [layer.weight.data.clone() for layer in self.layers]


__all__ = ["GraphQNN"]
