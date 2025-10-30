"""Hybrid classical estimator that blends feed‑forward regression with fidelity‑based graph analysis.

The class accepts an arbitrary layer architecture, trains a PyTorch network,
and can build a weighted graph of hidden activations using state fidelity.
A ``run`` method emulates a fully‑connected quantum layer by returning the
mean tanh activation of the final layer for a supplied sequence of angles.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import numpy as np
import torch
from torch import nn, optim, Tensor

__all__ = ["HybridEstimatorQNN"]


def _random_linear(in_features: int, out_features: int) -> Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)


def _state_fidelity(vec_a: Tensor, vec_b: Tensor) -> float:
    a_norm = vec_a / (torch.norm(vec_a) + 1e-12)
    b_norm = vec_b / (torch.norm(vec_b) + 1e-12)
    return float((a_norm @ b_norm).item() ** 2)


def _fidelity_adjacency(
    states: Sequence[Tensor], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
        fid = _state_fidelity(s_i, s_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


class HybridEstimatorQNN(nn.Module):
    """
    Classical hybrid estimator.

    Parameters
    ----------
    arch : Sequence[int]
        Layer sizes, e.g. ``[2, 8, 4, 1]``.
    use_graph : bool, default=True
        Whether to expose a ``graph`` method that constructs a fidelity graph
        of hidden activations.
    graph_threshold : float, default=0.9
        Threshold for edge creation in the fidelity graph.
    """

    def __init__(
        self,
        arch: Sequence[int],
        use_graph: bool = True,
        graph_threshold: float = 0.9,
    ) -> None:
        super().__init__()
        self.arch = list(arch)
        layers: List[nn.Module] = []
        for in_f, out_f in zip(self.arch[:-1], self.arch[1:]):
            layers.append(nn.Linear(in_f, out_f))
            layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers[:-1])  # drop last tanh for regression
        self.use_graph = use_graph
        self.graph_threshold = graph_threshold

    # ---------------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    # ---------------------------------------------------------------------------

    def train_on(
        self,
        dataset: List[Tuple[Tensor, Tensor]],
        lr: float = 1e-3,
        epochs: int = 200,
        batch_size: int = 32,
    ) -> None:
        """Simple MSE training loop."""
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for _ in range(epochs):
            for x, y in loader:
                optimizer.zero_grad()
                pred = self(x)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()

    # ---------------------------------------------------------------------------

    def graph(self, samples: int = 100, threshold: float | None = None) -> nx.Graph:
        """Return a weighted graph of hidden activations constructed from random samples."""
        if not self.use_graph:
            raise RuntimeError("Graph construction disabled in this instance.")
        threshold = threshold if threshold is not None else self.graph_threshold
        # Generate random inputs matching first layer size
        samples_tensor = torch.randn(samples, self.arch[0], dtype=torch.float32)
        activations = self._capture_activations(samples_tensor)
        return _fidelity_adjacency(activations, threshold)

    def _capture_activations(self, x: Tensor) -> List[Tensor]:
        """Return a list of hidden layer outputs for each sample."""
        activations: List[Tensor] = []
        current = x
        for layer in self.net:
            current = layer(current)
            if isinstance(layer, nn.Linear):
                activations.append(current.detach())
        return activations

    # ---------------------------------------------------------------------------

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Mimic a quantum fully‑connected layer.

        Thetas are interpreted as scalar inputs to a linear layer followed by tanh,
        then averaged across the batch to produce a single expectation value.
        """
        theta_tensor = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        with torch.no_grad():
            out = torch.tanh(self.net[0](theta_tensor)).mean(dim=0)
        return out.numpy()

    # ---------------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(arch={self.arch})"
