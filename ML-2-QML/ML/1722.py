import itertools
from typing import Iterable, List, Tuple, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import networkx as nx

Tensor = torch.Tensor

def _random_linear(in_features: int, out_features: int) -> Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    weights: List[Tensor] = []
    for in_features, out_features in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_features, out_features))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight

def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    stored: List[List[Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        stored.append(activations)
    return stored

def state_fidelity(a: Tensor, b: Tensor) -> float:
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)

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
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

class GraphQNN(nn.Module):
    """
    Hybrid graph neural network that learns from fidelity-based adjacency graphs.
    """
    def __init__(self, qnn_arch: List[int], hidden_dim: int = 32, num_layers: int = 2, conv_type: str = "gcn"):
        super().__init__()
        self.qnn_arch = qnn_arch
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.conv_type = conv_type
        self.convs = nn.ModuleList()
        in_dim = qnn_arch[-1]
        for _ in range(num_layers):
            if conv_type == "gcn":
                self.convs.append(GCNConv(in_dim, hidden_dim))
            else:
                self.convs.append(SAGEConv(in_dim, hidden_dim))
            in_dim = hidden_dim
        self.out_lin = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, data: Data) -> Tensor:
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        return self.out_lin(x)

    def train_graph(self, data_loader: DataLoader, epochs: int = 50, lr: float = 0.01):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        for epoch in range(epochs):
            total_loss = 0.0
            for data in data_loader:
                optimizer.zero_grad()
                out = self(data)
                target = torch.full_like(out, data.edge_weight.sum() / data.num_nodes)
                loss = loss_fn(out, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            # print(f"Epoch {epoch+1}/{epochs} Loss: {total_loss/len(data_loader):.4f}")

    def embed(self, data: Data) -> Tensor:
        self.eval()
        with torch.no_grad():
            return self(data)

__all__ = [
    "GraphQNN",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
