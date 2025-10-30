import torch
import torch.nn as nn
import networkx as nx
import itertools
from typing import Iterable, Sequence, Tuple, List

Tensor = torch.Tensor

def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a weight matrix of shape (out_features, in_features)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate (features, target) pairs for a linear target mapping."""
    dataset = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random linear network and a matching training set."""
    weights = [_random_linear(in_f, out_f) for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:])]
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight

def feedforward(qnn_arch: Sequence[int], weights: Sequence[Tensor], samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
    """Run a forward pass through a classical feed‑forward network."""
    stored = []
    for features, _ in samples:
        activations = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        stored.append(activations)
    return stored

def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Squared overlap between two vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)

def fidelity_adjacency(states: Sequence[Tensor], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
    """Create a weighted graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

class GraphQNNHybrid(nn.Module):
    """Classical graph neural network with learnable weights and graph regularization."""
    def __init__(self, qnn_arch: Sequence[int]):
        super().__init__()
        self.qnn_arch = list(qnn_arch)
        self.layers = nn.ModuleList([nn.Linear(in_f, out_f) for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:])])

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = torch.tanh(layer(x))
        return x

    def train_epoch(self,
                    data_loader: Iterable[Tuple[Tensor, Tensor]],
                    optimizer: torch.optim.Optimizer,
                    criterion=nn.MSELoss(),
                    graph: nx.Graph | None = None,
                    reg_weight: float = 0.0):
        """Run one epoch of gradient descent with optional graph regulariser."""
        self.train()
        total_loss = 0.0
        for features, target in data_loader:
            optimizer.zero_grad()
            output = self(features)
            loss = criterion(output, target)
            if graph is not None and reg_weight > 0.0:
                # Penalise distant nodes in the graph
                num_nodes = len(graph.nodes)
                idx = torch.arange(num_nodes, dtype=torch.float32)
                dist_matrix = torch.cdist(idx.unsqueeze(0), idx.unsqueeze(0))
                reg = reg_weight * torch.mean(dist_matrix * loss.unsqueeze(0))
                loss += reg
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(data_loader)

def train_graph_qnn(model: GraphQNNHybrid,
                    data_loader: Iterable[Tuple[Tensor, Tensor]],
                    epochs: int,
                    lr: float = 1e-3,
                    reg_weight: float = 0.0,
                    graph: nx.Graph | None = None):
    """Unified training loop for the hybrid graph‑neural‑network."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        loss = model.train_epoch(data_loader, optimizer, reg_weight=reg_weight, graph=graph)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")

__all__ = [
    "GraphQNNHybrid",
    "random_network",
    "random_training_data",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
    "train_graph_qnn",
]
