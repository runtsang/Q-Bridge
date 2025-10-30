"""
GraphQNN – Classical feed‑forward neural network with graph‑based regularisation.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Create a random weight matrix for a linear layer."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(
    weight: Tensor, samples: int
) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic data for a single‑layer network.

    The target is the linear map defined by ``weight``.
    """
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Return architecture, weights, training data and the target layer.

    The last layer is the target – it is used to generate the training data.
    """
    weights: List[Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight


class GraphQNN:
    """Classical feed‑forward neural network with optional graph‑based regularisation."""

    def __init__(self, arch: Sequence[int], device: str = "cpu"):
        self.arch = list(arch)
        self.device = device
        self.model = self._build_model()
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def _build_model(self) -> nn.Module:
        layers = []
        for in_f, out_f in zip(self.arch[:-1], self.arch[1:]):
            layers.append(nn.Linear(in_f, out_f))
            layers.append(nn.Tanh())
        return nn.Sequential(*layers).to(self.device)

    def _forward_with_activations(
        self, x: Tensor
    ) -> List[Tensor]:
        activations: List[Tensor] = [x]
        current = x
        for layer in self.model:
            current = layer(current)
            activations.append(current)
        return activations

    def build_graph_from_activations(
        self,
        activations: List[Tensor],
        threshold: float = 0.9,
    ) -> nx.Graph:
        """Create a graph where nodes are activations and edges are high‑fidelity pairs."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(activations)))
        for (i, a_i), (j, a_j) in itertools.combinations(enumerate(activations), 2):
            fid = self.state_fidelity(a_i, a_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
        return graph

    def state_fidelity(self, a: Tensor, b: Tensor) -> float:
        """Compute the squared overlap between two vectors."""
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float((a_norm @ b_norm).item() ** 2)

    def graph_loss(self, graph: nx.Graph) -> torch.Tensor:
        """Return a simple penalty proportional to the total edge weight."""
        if graph.number_of_edges() == 0:
            return torch.tensor(0.0, device=self.device)
        total_weight = sum(
            d.get("weight", 1.0) for _, _, d in graph.edges(data=True)
        )
        return torch.tensor(total_weight, device=self.device)

    def train(
        self,
        data: List[Tuple[Tensor, Tensor]],
        epochs: int = 10,
        lr: float = 0.01,
        graph_reg: float = 0.0,
        graph_threshold: float = 0.9,
    ) -> None:
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        for epoch in range(epochs):
            epoch_loss = 0.0
            for x, y in data:
                x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()
                out = self.model(x)
                loss = F.mse_loss(out, y)
                if graph_reg > 0.0:
                    activations = self._forward_with_activations(x)
                    graph = self.build_graph_from_activations(
                        activations, threshold=graph_threshold
                    )
                    loss += graph_reg * self.graph_loss(graph)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            # Uncomment to see progress
            # print(f"Epoch {epoch}: loss={epoch_loss/len(data):.4f}")

    def evaluate(
        self, data: List[Tuple[Tensor, Tensor]]
    ) -> float:
        self.model.eval()
        with torch.no_grad():
            loss = 0.0
            for x, y in data:
                x = x.to(self.device)
                y = y.to(self.device)
                out = self.model(x)
                loss += F.mse_loss(out, y, reduction="sum").item()
            return loss / len(data)


__all__ = [
    "GraphQNN",
    "random_network",
    "random_training_data",
]
