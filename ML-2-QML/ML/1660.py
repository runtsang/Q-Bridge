"""Python module defining GraphQNNGen with classical training and fidelity monitoring."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import torch

Tensor = torch.Tensor

@dataclass
class GraphQNNGen:
    """
    A hybrid classical graph neural network that tracks state fidelity.

    The network is a stack of linear layers with tanh activations.  The
    target transformation is a random linear map (the last layer).  During
    training the network is fitted to that target and a fidelity‑based
    adjacency graph of the final hidden states is built.  If the graph
    stops changing for a set number of epochs training halts early.
    """
    qnn_arch: Sequence[int]
    weights: List[Tensor]
    target: Tensor
    training_data: List[Tuple[Tensor, Tensor]]
    fidelity_graph: nx.Graph | None = None

    @staticmethod
    def _random_linear(in_features: int, out_features: int) -> Tensor:
        """Return a weight matrix of shape (out_features, in_features)."""
        return torch.randn(out_features, in_features, dtype=torch.float32, requires_grad=True)

    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int = 100) -> "GraphQNNGen":
        """Create a random network and matching training data."""
        weights = [_random_linear(in_, out_) for in_, out_ in zip(qnn_arch[:-1], qnn_arch[1:])]
        target = weights[-1]
        training_data = []
        for _ in range(samples):
            x = torch.randn(target.size(1), dtype=torch.float32)
            y = target @ x
            training_data.append((x, y))
        return GraphQNNGen(qnn_arch, weights, target, training_data)

    # ------------------------------------------------------------------ #
    # Forward pass
    # ------------------------------------------------------------------ #
    def _forward(self, x: Tensor) -> Tensor:
        cur = x
        for w in self.weights:
            cur = torch.tanh(w @ cur)
        return cur

    def feedforward(self, samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
        """Return activations of every layer for each sample."""
        all_acts: List[List[Tensor]] = []
        for x, _ in samples:
            acts = [x]
            cur = x
            for w in self.weights:
                cur = torch.tanh(w @ cur)
                acts.append(cur)
            all_acts.append(acts)
        return all_acts

    # ------------------------------------------------------------------ #
    # Fidelity utilities
    # ------------------------------------------------------------------ #
    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        """Squared overlap after normalisation."""
        a_n = a / (torch.norm(a) + 1e-12)
        b_n = b / (torch.norm(b) + 1e-12)
        return float((a_n @ b_n).item() ** 2)

    def build_fidelity_graph(self,
                             threshold: float,
                             secondary: float | None = None,
                             secondary_weight: float = 0.5) -> nx.Graph:
        """Construct a graph from the last‑layer states of the training data."""
        states = [acts[-1] for acts in self.feedforward(self.training_data)]
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, sa), (j, sb) in itertools.combinations(enumerate(states), 2):
            fid = self.state_fidelity(sa, sb)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        self.fidelity_graph = graph
        return graph

    # ------------------------------------------------------------------ #
    # Training routine
    # ------------------------------------------------------------------ #
    def train(self,
              epochs: int = 100,
              lr: float = 1e-3,
              fidelity_threshold: float = 0.99,
              patience: int = 5) -> None:
        """Train the network using MSE loss and early stopping based on fidelity graph."""
        opt = torch.optim.Adam(self.weights, lr=lr)
        best_graph = None
        no_improve = 0
        for epoch in range(1, epochs + 1):
            opt.zero_grad()
            loss = 0.0
            for x, y in self.training_data:
                out = self._forward(x)
                loss += torch.nn.functional.mse_loss(out, y)
            loss /= len(self.training_data)
            loss.backward()
            opt.step()

            # Update fidelity graph every few epochs
            if epoch % 5 == 0:
                graph = self.build_fidelity_graph(fidelity_threshold)
                if best_graph is not None and nx.is_isomorphic(graph, best_graph):
                    no_improve += 1
                    if no_improve >= patience:
                        print(f"Early stopping at epoch {epoch}")
                        break
                else:
                    best_graph = graph
                    no_improve = 0

    # ------------------------------------------------------------------ #
    # Utility
    # ------------------------------------------------------------------ #
    def evaluate(self, sample: Tensor) -> Tensor:
        """Forward a single sample through the network."""
        return self._forward(sample)

    def __repr__(self) -> str:
        return f"<GraphQNNGen arch={self.qnn_arch} samples={len(self.training_data)}>"

__all__ = ["GraphQNNGen"]
