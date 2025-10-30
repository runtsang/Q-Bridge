import itertools
from typing import Iterable, Sequence, List, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

Tensor = torch.Tensor

class GraphQNN:
    """Hybrid graph-based QNN generator with classical feedforward and fidelity adjacency.

    The class generates a random linear network, provides feedforward propagation,
    constructs a fidelity‑based adjacency graph, and offers a simple GNN predictor
    that can be trained with a mean‑squared‑error loss.
    """

    def __init__(self, qnn_arch: Sequence[int]):
        self.qnn_arch = list(qnn_arch)
        self.weights: List[Tensor] = [self._random_linear(in_f, out_f)
                                      for in_f, out_f in zip(self.qnn_arch[:-1], self.qnn_arch[1:])]

    @staticmethod
    def _random_linear(in_features: int, out_features: int) -> Tensor:
        """Return a random weight matrix of shape (out_features, in_features)."""
        return torch.randn(out_features, in_features, dtype=torch.float32)

    @staticmethod
    def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
        """Generate (feature, target) pairs for a linear target layer."""
        dataset: List[Tuple[Tensor, Tensor]] = []
        for _ in range(samples):
            features = torch.randn(weight.size(1))
            target = weight @ features
            dataset.append((features, target))
        return dataset

    def feedforward(self, samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
        """Compute activations for each sample across all layers."""
        outputs: List[List[Tensor]] = []
        for features, _ in samples:
            activations: List[Tensor] = [features]
            current = features
            for weight in self.weights:
                current = torch.tanh(weight @ current)
                activations.append(current)
            outputs.append(activations)
        return outputs

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        """Compute squared overlap between two pure states."""
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float(torch.dot(a_norm, b_norm).item() ** 2)

    @staticmethod
    def fidelity_adjacency(states: Sequence[Tensor], threshold: float,
                           *,
                           secondary: float | None = None,
                           secondary_weight: float = 0.5) -> nx.Graph:
        """Create a weighted adjacency graph from state fidelities."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNN.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    def train(self, dataset: List[Tuple[Tensor, Tensor]], lr: float = 0.01, epochs: int = 100) -> None:
        """Simple MSE training of the linear layers."""
        optim_sgd = optim.SGD(self.weights, lr=lr)
        for _ in range(epochs):
            for features, target in dataset:
                optim_sgd.zero_grad()
                output = self._forward_one(features)
                loss = F.mse_loss(output, target)
                loss.backward()
                optim_sgd.step()

    def _forward_one(self, features: Tensor) -> Tensor:
        """Forward a single sample through the network."""
        current = features
        for weight in self.weights:
            current = torch.tanh(weight @ current)
        return current

    def build_gnn_predictor(self, in_dim: int, hidden_dim: int, out_dim: int) -> nn.Module:
        """Return a simple two‑layer MLP as a placeholder GNN predictor."""
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

__all__ = ["GraphQNN"]
