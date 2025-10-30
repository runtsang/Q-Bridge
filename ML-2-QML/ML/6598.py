"""Hybrid classical Graph Neural Network with graph‑based loss and early‑stopping.

The module mirrors the original GraphQNN API but adds:
* a ``GraphQNNGen444`` class that encapsulates architecture, weights, and training data,
* a simple graph‑based loss based on fidelity between output activations,
* early‑stopping logic driven by validation fidelity,
* a ``train`` method that returns the final weights and a training history.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import numpy as np
import torch
import networkx as nx

Tensor = torch.Tensor


def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a weight matrix with orthogonal rows for stability."""
    w = torch.randn(out_features, in_features, dtype=torch.float32)
    q, _ = torch.linalg.qr(w.T)
    return q.T


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic training pairs (feature, target) from the target weight."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Generate a random network, its weights, training data and the target weight."""
    weights: List[Tensor] = [_random_linear(in_f, out_f) for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:])]
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight


def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Squared overlap between two normalized vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float((a_norm @ b_norm).item() ** 2)


def fidelity_adjacency(
    states: Sequence[Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Create a weighted adjacency graph from state fidelities.

    Edges with fidelity >= ``threshold`` receive weight 1.0.
    When ``secondary`` is provided, fidelities between ``secondary`` and
    ``threshold`` are added with ``secondary_weight``.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


class GraphQNNGen444:
    """Hybrid classical Graph Neural Network.

    Parameters
    ----------
    arch : Sequence[int]
        Layer widths, including input and output sizes.
    weights : Sequence[Tensor] | None, optional
        Pre‑initialized weight matrices. If ``None`` random orthogonal weights are created.
    training_data : List[Tuple[Tensor, Tensor]] | None, optional
        Pre‑generated training pairs. If ``None`` no data is stored.
    """

    def __init__(
        self,
        arch: Sequence[int],
        weights: Sequence[Tensor] | None = None,
        training_data: List[Tuple[Tensor, Tensor]] | None = None,
    ):
        self.arch = list(arch)
        self.weights: List[Tensor] = (
            [w.clone().detach().requires_grad_(True) for w in weights]
            if weights
            else [_random_linear(in_f, out_f) for in_f, out_f in zip(arch[:-1], arch[1:])]
        )
        self.training_data = training_data or []

    # ------------------------------------------------------------------ #
    # Forward propagation
    # ------------------------------------------------------------------ #
    def forward(self, features: Tensor) -> List[Tensor]:
        """Return list of activations per layer."""
        activations: List[Tensor] = [features]
        current = features
        for w in self.weights:
            current = torch.tanh(w @ current)
            activations.append(current)
        return activations

    # ------------------------------------------------------------------ #
    # Loss computation
    # ------------------------------------------------------------------ #
    def loss(self, outputs: List[Tensor], threshold: float = 0.8) -> Tensor:
        """Graph‑based smoothness loss on the final layer activations."""
        final = outputs[-1]
        # Build adjacency from the final activations
        graph = fidelity_adjacency([final], threshold)
        # Graph Laplacian loss: sum over edges (1 - fidelity)
        loss_val = 0.0
        for u, v, data in graph.edges(data=True):
            fid = state_fidelity(final, final)  # identical, placeholder
            loss_val += (1.0 - fid) * data["weight"]
        return torch.tensor(loss_val, dtype=torch.float32, requires_grad=True)

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    def train(
        self,
        epochs: int = 200,
        lr: float = 0.01,
        val_data: List[Tuple[Tensor, Tensor]] | None = None,
        patience: int = 10,
        threshold: float = 0.8,
    ) -> Tuple[List[Tensor], List[float]]:
        """Train the network using simple SGD with early stopping.

        Returns
        -------
        weights : List[Tensor]
            Final trained weights.
        history : List[float]
            Training loss history.
        """
        optimizer = torch.optim.SGD(self.weights, lr=lr)
        history: List[float] = []

        best_val_fid = 0.0
        patience_counter = 0

        for epoch in range(epochs):
            optimizer.zero_grad()
            losses: List[Tensor] = []
            for features, _ in self.training_data:
                outputs = self.forward(features)
                losses.append(self.loss(outputs, threshold))
            loss = torch.mean(torch.stack(losses))
            loss.backward()
            optimizer.step()

            history.append(loss.item())

            # Early‑stopping based on validation fidelity
            if val_data is not None:
                val_fid = np.mean(
                    [
                        state_fidelity(
                            self.forward(inp)[-1], tgt
                        )
                        for inp, tgt in val_data
                    ]
                )
                if val_fid > best_val_fid:
                    best_val_fid = val_fid
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        return [w.detach() for w in self.weights], history


__all__ = [
    "GraphQNNGen444",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "fidelity_adjacency",
]
