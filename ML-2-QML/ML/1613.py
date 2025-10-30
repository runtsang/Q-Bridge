import torch
import torch.nn as nn
import networkx as nx
import itertools
from typing import Iterable, Sequence, Tuple

Tensor = torch.Tensor

def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Return the squared magnitude of the inner product of two vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float((a_norm @ b_norm).item() ** 2)

def random_training_data(weight: Tensor, samples: int) -> list[Tuple[Tensor, Tensor]]:
    dataset: list[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    weights = [torch.randn(out, in_, dtype=torch.float32, requires_grad=False)
               for in_, out in zip(qnn_arch[:-1], qnn_arch[1:])]
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight

def feedforward(qnn_arch: Sequence[int], weights: Sequence[Tensor], samples: Iterable[Tuple[Tensor, Tensor]]) -> list[list[Tensor]]:
    stored: list[list[Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        for w in weights:
            current = torch.tanh(w @ current)
            activations.append(current)
        stored.append(activations)
    return stored

def fidelity_adjacency(states: Sequence[Tensor], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5):
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

class GraphQNN(nn.Module):
    """Graph‑structured neural network with trainable adjacency."""

    def __init__(self, arch: Sequence[int], adjacency_threshold: float = 0.5, adjacency_secondary: float | None = None):
        super().__init__()
        self.arch = list(arch)
        self.layers = nn.ModuleList([
            nn.Linear(in_f, out_f, bias=False)
            for in_f, out_f in zip(arch[:-1], arch[1:])
        ])
        self.adjacency_threshold = adjacency_threshold
        self.adjacency_secondary = adjacency_secondary
        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(len(arch[1:])))

    def forward(self, x: Tensor) -> list[Tensor]:
        activations: list[Tensor] = [x]
        for layer in self.layers:
            x = torch.tanh(layer(x))
            activations.append(x)
        return activations

    def update_adjacency(self) -> None:
        """Rebuild the adjacency graph from the current layer weights."""
        self.graph.clear()
        self.graph.add_nodes_from(range(len(self.arch[1:])))
        for layer in self.layers:
            w = layer.weight
            for i in range(w.size(0)):
                for j in range(w.size(1)):
                    fid = state_fidelity(w[i], w[j])
                    if fid >= self.adjacency_threshold:
                        self.graph.add_edge(i, j, weight=1.0)
                    elif self.adjacency_secondary is not None and fid >= self.adjacency_secondary:
                        self.graph.add_edge(i, j, weight=0.5)

    def train_step(self, data_loader, lr: float = 1e-3, epochs: int = 10):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        for epoch in range(epochs):
            for x, y in data_loader:
                optimizer.zero_grad()
                out = self.forward(x)
                loss = loss_fn(out[-1], y)
                loss.backward()
                optimizer.step()
            self.update_adjacency()

__all__ = [
    "GraphQNN",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
