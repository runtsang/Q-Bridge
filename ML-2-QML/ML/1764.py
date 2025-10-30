"""GraphQNNGen354: Classical graph neural network with extended capabilities.

This module retains the seed API while adding multi‑output layers,
mini‑batch training with Adam, and a simple graph‑based fidelity
regulariser.  The class exposes feedforward, random_network, and
random_training_data helpers that mirror the original functions.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple, Sequence

import itertools

import torch
import torch.nn as nn
import networkx as nx

Tensor = torch.Tensor


def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a randomly initialized weight matrix."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(
    weight: Tensor, samples: int
) -> List[Tuple[Tensor, Tensor]]:
    """Generate (x, y) pairs for a simple linear regression task."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        x = torch.randn(weight.size(1), dtype=torch.float32)
        y = weight @ x
        dataset.append((x, y))
    return dataset


def random_network(
    arch: Sequence[int], samples: int
) -> Tuple[Sequence[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
    """Create a random network and training data."""
    weights: List[Tensor] = [
        _random_linear(in_, out) for in_, out in zip(arch[:-1], arch[1:])
    ]
    target_weight = weights[-1]
    dataset = random_training_data(target_weight, samples)
    return arch, weights, dataset, target_weight


class GraphQNNGen354:
    """Classical graph‑neural‑network with multi‑output layers and training."""

    def __init__(self, arch: Sequence[int], device: torch.device | str = "cpu"):
        self.arch: Sequence[int] = tuple(arch)
        self.device = torch.device(device)
        self.layers: nn.ModuleList = nn.ModuleList(
            nn.Linear(in_features, out_features)
            for in_features, out_features in zip(arch[:-1], arch[1:])
        ).to(self.device)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, x: Tensor) -> List[Tensor]:
        """Return activations for every layer."""
        activations: List[Tensor] = [x]
        for layer in self.layers:
            x = torch.tanh(layer(x))
            activations.append(x)
        return activations

    # ------------------------------------------------------------------
    # Compatibility wrappers
    # ------------------------------------------------------------------
    def feedforward(
        self,
        samples: Iterable[Tuple[Tensor, Tensor]],
    ) -> List[List[Tensor]]:
        """Feedforward over a dataset, returning activations for each sample."""
        stored: List[List[Tensor]] = []
        for x, _ in samples:
            activations = self.forward(x)
            stored.append(activations)
        return stored

    # ------------------------------------------------------------------
    # Training routine
    # ------------------------------------------------------------------
    def train(
        self,
        dataset: Iterable[Tuple[Tensor, Tensor]],
        epochs: int = 200,
        lr: float = 1e-3,
        batch_size: int = 32,
    ) -> List[float]:
        """Mini‑batch Adam training with MSE loss."""
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        losses: List[float] = []

        data_list = list(dataset)

        for epoch in range(epochs):
            epoch_loss = 0.0
            torch.manual_seed(epoch)
            indices = torch.randperm(len(data_list))
            for start in range(0, len(data_list), batch_size):
                batch_indices = indices[start : start + batch_size]
                batch = [data_list[i] for i in batch_indices]
                xs = torch.stack([x for x, _ in batch], dim=0).to(self.device)
                ys = torch.stack([y for _, y in batch], dim=0).to(self.device)

                optimizer.zero_grad()
                activations = self.forward(xs)
                pred = activations[-1]
                loss = criterion(pred, ys)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            losses.append(epoch_loss / (len(data_list) / batch_size))
        return losses

    # ------------------------------------------------------------------
    # Helper static methods
    # ------------------------------------------------------------------
    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float(torch.dot(a_norm, b_norm).item() ** 2)

    @staticmethod
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
            fid = GraphQNNGen354.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    # ------------------------------------------------------------------
    # Parameter iterator
    # ------------------------------------------------------------------
    def parameters(self) -> Iterable[nn.Parameter]:
        """Yield all learnable parameters."""
        return self.layers.parameters()
