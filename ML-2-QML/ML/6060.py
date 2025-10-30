"""GraphQNN__gen480: Classical GNN with trainable convolutional layers and synthetic data generation."""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn

Tensor = torch.Tensor


class GraphQNN__gen480:
    """A lightweight, fully‑trainable graph neural network that mirrors the original
    seed but adds trainable graph‑convolutional layers and a simple linear
    regression data generator for quick experimentation.

    Parameters
    ----------
    arch : Sequence[int]
        Layer sizes, e.g. ``[4, 8, 4]`` creates a network with an 4‑dimensional input,
        one hidden layer of 8 units and a 4‑dimensional output.
    """

    def __init__(self, arch: Sequence[int]):
        self.arch = list(arch)
        self.layers: nn.ModuleList = nn.ModuleList()
        for in_f, out_f in zip(self.arch[:-1], self.arch[1:]):
            self.layers.append(nn.Linear(in_f, out_f))
        self.activation = nn.Tanh

    # -------------------------------------------------------------------------
    # Data generation helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _random_linear(in_features: int, out_features: int) -> Tensor:
        """Return a random weight matrix of shape (out, in)."""
        return torch.randn(out_features, in_features, dtype=torch.float32)

    @classmethod
    def random_network(
        cls,
        arch: Sequence[int],
        samples: int,
        *,
        seed: int | None = None,
    ) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
        """
        Generate a random network, synthetic training data and the target weight
        matrix that the last layer is supposed to learn.

        Returns
        -------
        arch,
        weights,
        training_data,
        target_weight
        """
        rng = np.random.default_rng(seed)
        weights = [
            torch.tensor(rng.standard_normal((out, inp)), dtype=torch.float32)
            for inp, out in zip(arch[:-1], arch[1:])
        ]
        target_weight = weights[-1]
        training_data = cls.random_training_data(target_weight, samples, rng=rng)
        return list(arch), weights, training_data, target_weight

    @staticmethod
    def random_training_data(
        target_weight: Tensor,
        samples: int,
        *,
        rng: np.random.Generator | None = None,
    ) -> List[Tuple[Tensor, Tensor]]:
        """
        Create a synthetic linear regression dataset.

        Parameters
        ----------
        target_weight : Tensor
            The weight matrix that the network is expected to learn.
        samples : int
            Number of (feature, target) pairs to generate.
        rng : np.random.Generator, optional
            Random number generator for reproducibility.

        Returns
        -------
        List[(features, target)]
        """
        rng = rng or np.random.default_rng()
        features = torch.tensor(
            rng.standard_normal((samples, target_weight.size(1))), dtype=torch.float32
        )
        target = features @ target_weight.t()
        return list(zip(features, target))

    # -------------------------------------------------------------------------
    # Forward propagation
    # -------------------------------------------------------------------------
    def feedforward(
        self, samples: Iterable[Tuple[Tensor, Tensor]]
    ) -> List[List[Tensor]]:
        """
        Run a batch of samples through the network and return a list of
        activation sequences for each example.

        Each returned inner list contains the input and the activation of each
        subsequent layer.
        """
        stored: List[List[Tensor]] = []
        for features, _ in samples:
            activations: List[Tensor] = [features]
            current = features
            for layer in self.layers:
                current = self.activation(layer(current))
                activations.append(current)
            stored.append(activations)
        return stored

    # -------------------------------------------------------------------------
    # Fidelity helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        """
        Cosine similarity squared between two vectors, analogous to quantum
        state fidelity for pure states.
        """
        a_norm = a / (torch.norm(a, dim=-1, keepdim=True) + 1e-12)
        b_norm = b / (torch.norm(b, dim=-1, keepdim=True) + 1e-12)
        return float((a_norm @ b_norm.t()).item() ** 2)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """
        Build a weighted graph from state fidelities.  Edges with fidelity
        >= ``threshold`` receive weight 1; those in the secondary band receive
        ``secondary_weight``.
        """
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNN__gen480.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    # -------------------------------------------------------------------------
    # Training utilities
    # -------------------------------------------------------------------------
    def train(
        self,
        training_data: Iterable[Tuple[Tensor, Tensor]],
        lr: float = 1e-3,
        epochs: int = 200,
        verbose: bool = False,
    ) -> None:
        """
        Simple SGD training loop for the network.

        Parameters
        ----------
        training_data : iterable of (input, target) pairs
        lr : learning rate
        epochs : number of passes over the data
        verbose : if True, prints loss every 10 epochs
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            for features, target in training_data:
                optimizer.zero_grad()
                out = self.forward(features)
                loss = loss_fn(out, target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch:3d} – loss: {epoch_loss / len(training_data):.6f}")

    def forward(self, x: Tensor) -> Tensor:
        """Convenience wrapper for a single forward pass."""
        for layer in self.layers:
            x = self.activation(layer(x))
        return x

    # -------------------------------------------------------------------------
    # Convenience helpers
    # -------------------------------------------------------------------------
    def parameters(self):
        """Yield all learnable parameters of the network."""
        return self.layers.parameters()
