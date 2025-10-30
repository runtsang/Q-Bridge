from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import torch

Tensor = torch.Tensor


class GraphQNN:
    """A classical graph‑neural‑network style model with autograd support."""

    def __init__(
        self,
        architecture: Sequence[int],
        device: torch.device | str = torch.device("cpu"),
        lr: float = 1e-3,
    ):
        self.arch = list(architecture)
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.weights: List[Tensor] = [
            torch.randn(out, in_, dtype=torch.float32, requires_grad=True, device=self.device)
            for in_, out in zip(self.arch[:-1], self.arch[1:])
        ]
        self.optimizer = torch.optim.Adam(self.weights, lr=lr)
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x: Tensor) -> List[Tensor]:
        """Return activations for each layer."""
        activations = [x]
        current = x
        for w in self.weights:
            current = torch.tanh(w @ current)
            activations.append(current)
        return activations

    def train(
        self,
        data: List[Tuple[Tensor, Tensor]],
        epochs: int = 100,
        verbose: bool = True,
    ) -> None:
        """Gradient‑based training loop."""
        for epoch in range(epochs):
            epoch_loss = 0.0
            for x, y in data:
                self.optimizer.zero_grad()
                output = self.forward(x)[-1]
                loss = self.loss_fn(output, y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            if verbose:
                print(f"Epoch {epoch:03d}: loss={epoch_loss / len(data):.6f}")

    def state_fidelity(self, a: Tensor, b: Tensor) -> float:
        """Squared overlap between two torch vectors."""
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
        """Build a weighted graph from pairwise fidelities."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
            fid = self.state_fidelity(s_i, s_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    @staticmethod
    def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
        """Generate synthetic data for a linear target mapping."""
        dataset: List[Tuple[Tensor, Tensor]] = []
        for _ in range(samples):
            features = torch.randn(weight.size(1), dtype=torch.float32)
            target = weight @ features
            dataset.append((features, target))
        return dataset

    @staticmethod
    def random_network(arch: Sequence[int], samples: int):
        """Create a random weight list and corresponding training data."""
        weights: List[Tensor] = [
            torch.randn(out, in_, dtype=torch.float32, requires_grad=True)
            for in_, out in zip(arch[:-1], arch[1:])
        ]
        target_weight = weights[-1]
        training_data = GraphQNN.random_training_data(target_weight, samples)
        return list(arch), weights, training_data, target_weight
