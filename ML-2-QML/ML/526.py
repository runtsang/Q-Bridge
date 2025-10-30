"""
Hybrid Graph Neural Network – Classical implementation.

This module extends the original GraphQNN API by adding:
* A full training pipeline that optimizes all weight matrices with Adam.
* A helper that builds a weighted graph of weight (unitary) distances.
* A convenience wrapper `run_full` that stitches together network generation,
  training and adjacency construction.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim

Tensor = torch.Tensor


class GraphQNN__gen167:
    """
    Classical graph neural network with extended utilities.
    """

    def __init__(self, arch: Sequence[int], device: str = "cpu"):
        self.arch = arch
        self.device = device

    @staticmethod
    def _random_linear(in_features: int, out_features: int) -> Tensor:
        """Return a tensor of shape (out_features, in_features) with standard normal entries."""
        return torch.randn(out_features, in_features, dtype=torch.float32)

    @staticmethod
    def random_network(
        arch: Sequence[int], samples: int
    ) -> Tuple[Sequence[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
        """
        Generate a random weight network and a training set for the target
        final layer. The returned target_weight is the last weight matrix.
        """
        weights: List[Tensor] = []
        for in_f, out_f in zip(arch[:-1], arch[1:]):
            weights.append(GraphQNN__gen167._random_linear(in_f, out_f))
        target_weight = weights[-1]
        training_data = GraphQNN__gen167.random_training_data(target_weight, samples)
        return arch, weights, training_data, target_weight

    @staticmethod
    def random_training_data(
        weight: Tensor, samples: int
    ) -> List[Tuple[Tensor, Tensor]]:
        """
        Produce a list of (feature, target) tuples where the target is
        computed by applying the supplied weight matrix to a random feature vector.
        """
        dataset: List[Tuple[Tensor, Tensor]] = []
        for _ in range(samples):
            features = torch.randn(weight.size(1), dtype=torch.float32)
            target = weight @ features
            dataset.append((features, target))
        return dataset

    def feedforward(
        self,
        weights: Sequence[Tensor],
        samples: Iterable[Tuple[Tensor, Tensor]],
    ) -> List[List[Tensor]]:
        """
        Forward pass through the network.  All intermediate activations are
        collected and returned as a list of lists.
        """
        stored: List[List[Tensor]] = []
        for features, _ in samples:
            activations = [features]
            current = features
            for weight in weights:
                current = torch.tanh(weight @ current)
                activations.append(current)
            stored.append(activations)
        return stored

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        """
        Return the absolute squared overlap between two vectors.
        """
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
        """
        Create a weighted graph from state fidelities.
        """
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNN__gen167.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    @staticmethod
    def unitary_distance_graph(
        weights: Sequence[Tensor],
        threshold: float = 0.1,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """
        Build a graph where nodes are weight matrices.  Edge weights are the
        normalized Frobenius distance between matrices.
        """
        graph = nx.Graph()
        graph.add_nodes_from(range(len(weights)))

        def frob_dist(A: Tensor, B: Tensor) -> float:
            return torch.norm(A - B, p="fro").item()

        for (i, w_i), (j, w_j) in itertools.combinations(enumerate(weights), 2):
            dist = frob_dist(w_i, w_j)
            if dist >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and dist >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    @staticmethod
    def run_full(
        arch: Sequence[int],
        samples: int,
        epochs: int = 20,
        lr: float = 0.01,
        device: str = "cpu",
    ) -> Tuple[List[Tensor], nx.Graph]:
        """
        Convenience wrapper that:
        1. Generates a random network and training data.
        2. Trains all weight matrices with Adam and MSE loss.
        3. Builds a fidelity adjacency graph from the final activations.
        """
        arch, weights, training_data, _ = GraphQNN__gen167.random_network(arch, samples)

        # Convert weights to trainable parameters
        params = [torch.nn.Parameter(w.to(device)) for w in weights]
        optimizer = optim.Adam(params, lr=lr)
        loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            epoch_loss = 0.0
            for features, target in training_data:
                features = features.to(device)
                target = target.to(device)

                # Forward
                current = features
                for param in params:
                    current = torch.tanh(param @ current)

                loss = loss_fn(current, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

        # Evaluate final activations on the training set
        final_states = []
        for features, _ in training_data:
            current = features
            for param in params:
                current = torch.tanh(param @ current)
            final_states.append(current.detach().cpu())

        adjacency = GraphQNN__gen167.fidelity_adjacency(final_states, threshold=0.8)
        trained_weights = [p.detach().cpu() for p in params]
        return trained_weights, adjacency


__all__ = ["GraphQNN__gen167"]
