import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import itertools
from typing import List, Tuple, Sequence, Iterable

Tensor = torch.Tensor

class GraphQNN:
    """
    Classical graph neural network with a feed‑forward architecture.
    """
    def __init__(self, qnn_arch: Sequence[int], device: str = "cpu"):
        self.arch = list(qnn_arch)
        self.device = device
        self.layers: nn.ModuleList = nn.ModuleList()
        for in_f, out_f in zip(self.arch[:-1], self.arch[1:]):
            self.layers.append(nn.Linear(in_f, out_f, bias=False))
        self.to(self.device)

    def feedforward(self, x: Tensor) -> List[Tensor]:
        """
        Return the list of activations for all layers.
        """
        activations: List[Tensor] = [x]
        current = x
        for layer in self.layers:
            current = torch.tanh(layer(current))
            activations.append(current)
        return activations

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        """
        Squared overlap between two pure states.
        """
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float((a_norm @ b_norm).item() ** 2)

    @staticmethod
    def fidelity_adjacency(states: Sequence[Tensor], threshold: float,
                           *, secondary: float | None = None,
                           secondary_weight: float = 0.5) -> nx.Graph:
        """
        Build a weighted graph from state fidelities.
        """
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNN.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    @staticmethod
    def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
        """
        Generate synthetic training data from a linear map.
        """
        data: List[Tuple[Tensor, Tensor]] = []
        for _ in range(samples):
            features = torch.randn(weight.size(1), dtype=torch.float32)
            target = weight @ features
            data.append((features, target))
        return data

    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
        """
        Create a random network and synthetic dataset.
        """
        weights: List[Tensor] = []
        for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
            w = torch.randn(out_f, in_f, dtype=torch.float32)
            weights.append(w)
        target_weight = weights[-1]
        training_data = GraphQNN.random_training_data(target_weight, samples)
        return list(qnn_arch), weights, training_data, target_weight

    def train(self, data: List[Tuple[Tensor, Tensor]], lr: float = 1e-3, epochs: int = 200) -> None:
        """
        Train the network on synthetic data.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        for _ in range(epochs):
            for x, y in data:
                optimizer.zero_grad()
                out = self.feedforward(x)[-1]
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
