from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


class GraphQNN(nn.Module):
    """Hybrid classical graph neural network with optional quantum-inspired layers."""

    def __init__(self, arch: Sequence[int], weights: Sequence[Tensor] | None = None):
        super().__init__()
        self.arch = list(arch)
        if weights is None:
            self.weights = nn.ParameterList(
                [nn.Parameter(torch.randn(out, in_)) for in_, out in zip(self.arch[:-1], self.arch[1:])]
            )
        else:
            self.weights = nn.ParameterList([nn.Parameter(w) for w in weights])

    def forward(self, x: Tensor) -> Tensor:
        out = x
        for weight in self.weights:
            out = torch.tanh(weight @ out)
        return out

    def feedforward(
        self, samples: Iterable[Tuple[Tensor, Tensor]]
    ) -> List[List[Tensor]]:
        stored: List[List[Tensor]] = []
        for x, _ in samples:
            activations = [x]
            current = x
            for weight in self.weights:
                current = torch.tanh(weight @ current)
                activations.append(current)
            stored.append(activations)
        return stored

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float((a_norm @ b_norm).item() ** 2)

    def fidelity_adjacency(
        self,
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
            fid = self.state_fidelity(a, b)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    @staticmethod
    def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
        dataset: List[Tuple[Tensor, Tensor]] = []
        for _ in range(samples):
            x = torch.randn(weight.size(1))
            y = weight @ x
            dataset.append((x, y))
        return dataset

    @staticmethod
    def random_network(arch: Sequence[int], samples: int):
        weights = [torch.randn(out, in_) for in_, out in zip(arch[:-1], arch[1:])]
        target_weight = weights[-1]
        training_data = GraphQNN.random_training_data(target_weight, samples)
        return GraphQNN(arch, weights), training_data, target_weight

    def train_random(self, epochs: int = 10, lr: float = 0.01, batch_size: int = 32):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for epoch in range(epochs):
            self.train()
            data = self.random_training_data(self.weights[-1], batch_size)
            for x, y in data:
                optimizer.zero_grad()
                pred = self.forward(x)
                loss = F.mse_loss(pred, y)
                loss.backward()
                optimizer.step()
        return self

    def graph_optimizer(self, graph: nx.Graph, max_cluster_size: int = 5) -> dict[int, int]:
        """Simple greedy clustering based on adjacency."""
        clusters: dict[int, int] = {}
        cluster_id = 0
        visited: set[int] = set()
        for node in graph.nodes():
            if node in visited:
                continue
            cluster: set[int] = {node}
            for neighbor in graph.neighbors(node):
                if neighbor not in visited and len(cluster) < max_cluster_size:
                    cluster.add(neighbor)
            for n in cluster:
                visited.add(n)
                clusters[n] = cluster_id
            cluster_id += 1
        return clusters
