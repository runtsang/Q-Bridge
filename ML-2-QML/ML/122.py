import itertools
from typing import Iterable, Sequence, List, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor

def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix of shape (out_features, in_features)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic training data for a linear target."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random classical MLP with the given architecture."""
    weights: List[Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight

class GraphQNN(nn.Module):
    """Classical graph neural network trainer.

    The network is a simple feed‑forward MLP with tanh activations and an optional
    dropout regulariser applied after each linear layer.
    """

    def __init__(self, qnn_arch: Sequence[int], dropout: float = 0.0, device: str = "cpu"):
        super().__init__()
        self.arch = list(qnn_arch)
        self.dropout = dropout
        self.device = device
        self.layers = nn.ModuleList()
        for in_f, out_f in zip(self.arch[:-1], self.arch[1:]):
            self.layers.append(nn.Linear(in_f, out_f))
        self.activation = nn.Tanh()
        self.drop_layer = nn.Dropout(p=self.dropout)
        self.to(self.device)

    def forward(self, x: Tensor) -> List[Tensor]:
        """Return a list of activations for each layer."""
        activations = [x]
        current = x
        for layer in self.layers:
            current = self.activation(layer(current))
            if self.dropout > 0.0:
                current = self.drop_layer(current)
            activations.append(current)
        return activations

    def feedforward(self, samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
        """Run the network on a batch of samples and return activations."""
        all_activations: List[List[Tensor]] = []
        for features, _ in samples:
            all_activations.append(self.forward(features))
        return all_activations

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        """Squared overlap between two classical vectors."""
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float(torch.dot(a_norm, b_norm).item() ** 2)

    @staticmethod
    def fidelity_adjacency(states: Sequence[Tensor], threshold: float,
                           *, secondary: float | None = None,
                           secondary_weight: float = 0.5) -> nx.Graph:
        """Build a weighted graph from pairwise fidelities."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNN.state_fidelity(s_i, s_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    def train(self, dataset: Iterable[Tuple[Tensor, Tensor]],
              lr: float = 1e-3, epochs: int = 10,
              loss_fn: nn.Module = nn.MSELoss()) -> None:
        """Train the network on the given dataset."""
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for _ in range(epochs):
            for x, y in dataset:
                optimizer.zero_grad()
                out = self.forward(x)[-1]
                loss = loss_fn(out, y)
                loss.backward()
                optimizer.step()

    @staticmethod
    def compare_models(a: "GraphQNN", b: "GraphQNN",
                       threshold: float = 0.8,
                       *, secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
        """Compare the last‑layer weight matrices of two models."""
        weights_a = [w.detach() for w in a.layers[-1].weight]
        graph = nx.Graph()
        graph.add_nodes_from(range(len(weights_a)))
        for (i, w_i), (j, w_j) in itertools.combinations(enumerate(weights_a), 2):
            fid = GraphQNN.state_fidelity(w_i, w_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

__all__ = [
    "GraphQNN",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "fidelity_adjacency",
]
