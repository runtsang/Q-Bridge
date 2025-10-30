import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import itertools
from typing import List, Tuple, Iterable, Sequence

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

class MessagePassingGNN(nn.Module):
    """
    Lightweight GNN that mirrors the classical feed‑forward architecture.
    """
    def __init__(self, arch: Sequence[int]):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(in_f, out_f) for in_f, out_f in zip(arch[:-1], arch[1:])]
        )
        self.activation = nn.Tanh()

    def forward(self, x: Tensor) -> Tensor:
        out = x
        for lin in self.layers:
            out = self.activation(lin(out))
        return out

class SharedGraphQNN(nn.Module):
    """
    Hybrid classical‑quantum graph neural network.
    The classical part is a MessagePassingGNN; the quantum part is a
    parameter‑shiftable linear embedding that can be used as a kernel.
    """
    def __init__(self, arch: Sequence[int], device: str = "cpu"):
        super().__init__()
        self.gnn = MessagePassingGNN(arch).to(device)
        # Quantum embedding parameters: a linear map from input to output
        self.q_params = nn.Parameter(torch.randn(arch[-1], arch[0], dtype=torch.float32))
        self.device = device

    def quantum_embedding(self, x: Tensor) -> Tensor:
        """
        Simple linear embedding that can be differentiated via
        automatic differentiation.
        """
        return torch.tanh(self.q_params @ x)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the classical GNN.
        """
        return self.gnn(x)

    def fidelity_graph(self, states: List[Tensor], threshold: float) -> nx.Graph:
        """
        Construct a graph from state fidelities.
        """
        return fidelity_adjacency(states, threshold)

    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int):
        return random_network(qnn_arch, samples)

    @staticmethod
    def random_training_data(weight: Tensor, samples: int):
        return random_training_data(weight, samples)

    @staticmethod
    def feedforward(
        qnn_arch: Sequence[int],
        weights: Sequence[Tensor],
        samples: Iterable[Tuple[Tensor, Tensor]],
    ) -> List[List[Tensor]]:
        return feedforward(qnn_arch, weights, samples)
