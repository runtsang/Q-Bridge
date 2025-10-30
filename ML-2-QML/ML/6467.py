import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

Tensor = torch.Tensor

class GraphQNN(nn.Module):
    """
    Classical graph‑based neural network that mirrors the quantum interface.
    Architecture is a list of layer widths.
    """

    def __init__(self, arch: Sequence[int], device: torch.device | str = "cpu"):
        super().__init__()
        self.arch = list(arch)
        self.device = torch.device(device)
        self.layers: nn.ModuleList = nn.ModuleList(
            [nn.Linear(in_f, out_f) for in_f, out_f in zip(arch[:-1], arch[1:])]
        )
        self.to(self.device)

    def forward(self, x: Tensor) -> List[Tensor]:
        activations: List[Tensor] = [x]
        current = x
        for layer in self.layers:
            current = torch.tanh(layer(current))
            activations.append(current)
        return activations

    @staticmethod
    def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
        """Generate (x, y) pairs where y = W x."""
        dataset: List[Tuple[Tensor, Tensor]] = []
        for _ in range(samples):
            x = torch.randn(weight.size(1), dtype=weight.dtype, device=weight.device)
            y = weight @ x
            dataset.append((x, y))
        return dataset

    def random_network(self, samples: int = 100) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
        """Create a random network and synthetic training data."""
        weights = [layer.weight.data.clone() for layer in self.layers]
        target_weight = weights[-1]
        training_data = self.random_training_data(target_weight, samples)
        return self.arch, weights, training_data, target_weight

    def fidelity_adjacency(
        self,
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Construct a weighted graph from state fidelities."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
            fid = torch.dot(a, b) ** 2
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    def train(
        self,
        training_data: Iterable[Tuple[Tensor, Tensor]],
        epochs: int = 10,
        lr: float = 1e-3,
        verbose: bool = False,
    ) -> List[float]:
        """Simple mean‑squared‑error training loop."""
        optimizer = optim.Adam(self.parameters(), lr=lr)
        losses: List[float] = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            for x, y_true in training_data:
                optimizer.zero_grad()
                _, y_pred = self.forward(x)
                loss = F.mse_loss(y_pred, y_true)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= len(training_data)
            losses.append(epoch_loss)
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs} – loss: {epoch_loss:.4f}")
        return losses
