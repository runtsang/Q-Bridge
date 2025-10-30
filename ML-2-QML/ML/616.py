"""GraphQNN__gen265 – Classical GNN with hybrid loss and variational support.

The new :class:`GraphQNN__gen265` class mirrors the original seed but now
supports a *two‑stage* training pipeline:
  * A lightweight PyTorch linear stack that learns a weight matrix
    matching the target quantum unitary.
  * An optional *quantum fidelity loss* that can be plugged into any
    training loop.
The code is fully importable and ready for downstream experiments.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import torch
from torch import nn

Tensor = torch.Tensor


class GraphQNN__gen265:
    """Hybrid classical–quantum graph neural network.

    The architecture is a sequence of layer sizes.  Each layer is a
    :class:`torch.nn.Linear` followed by a tanh activation.  The class
    exposes both pure classical operations (feedforward, random_network)
    and a *hybrid loss* that can be used to train the network to
    approximate a target quantum unitary.
    """

    def __init__(self, arch: Sequence[int]) -> None:
        self.arch = list(arch)
        self.layers: nn.ModuleList = nn.ModuleList()
        for in_f, out_f in zip(self.arch[:-1], self.arch[1:]):
            self.layers.append(nn.Linear(in_f, out_f, bias=False))
        self.device = torch.device("cpu")
        self.to(self.device)

    def to(self, device: torch.device) -> None:
        self.device = device
        self.layers.to(device)

    # ------------------------------------------------------------------
    #  Classical forward pass
    # ------------------------------------------------------------------
    def feedforward(self, features: Tensor) -> List[Tensor]:
        """Return activations for each layer.

        Args:
            features: Input feature vector of shape (in_features,).
        """
        activations: List[Tensor] = [features]
        current = features
        for layer in self.layers:
            current = torch.tanh(layer(current))
            activations.append(current)
        return activations

    # ------------------------------------------------------------------
    #  Random data generation utilities
    # ------------------------------------------------------------------
    @staticmethod
    def random_training_data(
        weight: Tensor, samples: int
    ) -> List[Tuple[Tensor, Tensor]]:
        """Generate synthetic (x, y) pairs where y = weight @ x."""
        dataset: List[Tuple[Tensor, Tensor]] = []
        in_dim = weight.size(1)
        for _ in range(samples):
            x = torch.randn(in_dim, dtype=torch.float32)
            y = weight @ x
            dataset.append((x, y))
        return dataset

    @staticmethod
    def random_network(
        arch: Sequence[int], samples: int
    ) -> Tuple[Sequence[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
        """Create a random linear network and a target weight for the last layer."""
        weights: List[Tensor] = []
        for in_f, out_f in zip(arch[:-1], arch[1:]):
            weights.append(torch.randn(out_f, in_f, dtype=torch.float32))
        target_weight = weights[-1]
        training_data = GraphQNN__gen265.random_training_data(target_weight, samples)
        return list(arch), weights, training_data, target_weight

    # ------------------------------------------------------------------
    #  Fidelity helpers
    # ------------------------------------------------------------------
    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        """Squared overlap of two state vectors."""
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float(torch.dot(a_norm, b_norm).item() ** 2)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[Tensor], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5
    ) -> nx.Graph:
        """Build a weighted graph from state fidelities."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNN__gen265.state_fidelity(a, b)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    # ------------------------------------------------------------------
    #  Hybrid loss
    # ------------------------------------------------------------------
    @staticmethod
    def hybrid_loss(
        pred: Tensor, target: Tensor, fid_weight: float = 0.1
    ) -> Tensor:
        """Combine MSE with a fidelity penalty."""
        mse = nn.functional.mse_loss(pred, target, reduction="mean")
        fid = GraphQNN__gen265.state_fidelity(pred, target)
        return mse + fid_weight * (1 - fid)

    # ------------------------------------------------------------------
    #  Training routine (classical only)
    # ------------------------------------------------------------------
    def train_classical(
        self, data: Iterable[Tuple[Tensor, Tensor]], lr: float = 1e-3, epochs: int = 200
    ) -> None:
        """Simple SGD training to match the target weight."""
        params = [p for layer in self.layers for p in layer.parameters()]
        optimizer = torch.optim.Adam(params, lr=lr)
        for _ in range(epochs):
            for x, y in data:
                optimizer.zero_grad()
                out = self.feedforward(x)[-1]
                loss = GraphQNN__gen265.hybrid_loss(out, y)
                loss.backward()
                optimizer.step()


__all__ = [
    "GraphQNN__gen265",
]
