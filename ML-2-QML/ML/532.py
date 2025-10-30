"""Hybrid Graph Neural Network with classical post‑processing.

This module extends the original GraphQNN helpers by adding a trainable
linear head that maps the last quantum‑like activation to a scalar
target.  The class ``HybridGraphQNN`` can be trained on a CPU‑only
simulator (scipy) and optionally back‑propagates the loss into the
quantum‑like weight matrices.

The public API mirrors the seed code: ``random_network``,
``random_training_data``, ``feedforward``, ``state_fidelity`` and
``fidelity_adjacency`` remain available as standalone functions.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
# Helper functions (original seed logic with small extensions)
# --------------------------------------------------------------------------- #

def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix with shape (out_features, in_features)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate ``samples`` inputs and targets from a fixed linear mapping."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(
    qnn_arch: Sequence[int], samples: int
) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
    """Return a full random network description and training data."""
    # 1. random linear weights for each layer
    weights: List[Tensor] = []
    for in_, out_ in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_, out_))
    # 2. target weight
    target_weight = weights[-1]
    # 3. training data
    seed_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, seed_data, target_weight

def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Compute layer‑wise activations for each sample as in the seed."""
    stored: List[List[Tensor]] = []
    for features, _ in samples:
        activations: List[Tensor] = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        stored.append(activations)
    return stored

def state_fidelity(a: Tensor, b: Tensor) -> float:
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)

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
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

# --------------------------------------------------------------------------- #
# HybridGraphQNN class
# --------------------------------------------------------------------------- #

class HybridGraphQNN:
    """A hybrid graph neural network with a classical linear head.

    Parameters
    ----------
    qnn_arch : Sequence[int]
        Architecture of the quantum‑like network: a list of layer widths.
    device : torch.device | str, optional
        Torch device on which the network operates.  Defaults to ``cpu``.
    head_dim : int, optional
        Output dimension of the classical head.  Defaults to 1.
    train_circuit : bool, optional
        If ``True`` the weight matrices are treated as learnable
        parameters and updated during training.
    """
    def __init__(
        self,
        qnn_arch: Sequence[int],
        device: torch.device | str = "cpu",
        head_dim: int = 1,
        train_circuit: bool = False,
    ) -> None:
        self.arch = list(qnn_arch)
        self.device = torch.device(device)
        self.train_circuit = train_circuit

        # Create network weights
        self.weights: List[Tensor] = [
            nn.Parameter(_random_linear(in_, out_).to(self.device))
            for in_, out_ in zip(self.arch[:-1], self.arch[1:])
        ]

        # Classical head
        self.head = nn.Linear(self.arch[-1], head_dim, bias=True).to(self.device)

        # Optimizer placeholder (set in ``fit``)
        self.optimizer: optim.Optimizer | None = None

    # ----------------------------------------------------------------------- #
    # Forward pass (used by predict and training)
    # ----------------------------------------------------------------------- #
    def _forward(self, x: Tensor) -> Tensor:
        """Return the final activation after the quantum‑like layers."""
        current = x.to(self.device)
        for weight in self.weights:
            current = torch.tanh(weight @ current)
        return current

    # ----------------------------------------------------------------------- #
    # Public API
    # ----------------------------------------------------------------------- #
    def predict(self, X: Iterable[Tensor]) -> Tensor:
        """Apply the network to ``X`` and return the head output."""
        self.eval()
        with torch.no_grad():
            final_activations = [self._forward(x) for x in X]
            stack = torch.stack(final_activations)
            return self.head(stack)

    def eval(self) -> None:
        """Set the network to evaluation mode."""
        self.head.eval()
        for w in self.weights:
            w.requires_grad = False

    def train(self) -> None:
        """Set the network to training mode."""
        self.head.train()
        if self.train_circuit:
            for w in self.weights:
                w.requires_grad = True

    # ----------------------------------------------------------------------- #
    # Training loop
    # ----------------------------------------------------------------------- #
    def fit(
        self,
        training_data: Iterable[Tuple[Tensor, Tensor]],
        epochs: int = 100,
        lr: float = 0.01,
        weight_decay: float = 0.0,
    ) -> None:
        """Train the head (and optionally the circuit) on ``training_data``."""
        self.train()
        # Build optimizer
        params = list(self.head.parameters())
        if self.train_circuit:
            params += [w for w in self.weights if w.requires_grad]
        self.optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            epoch_loss = 0.0
            for x, y in training_data:
                self.optimizer.zero_grad()
                pred = self.head(self._forward(x))
                loss = loss_fn(pred, y.to(self.device))
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            if (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch+1}/{epochs} – loss: {epoch_loss/len(training_data):.4f}")

    # ----------------------------------------------------------------------- #
    # Utility methods (mirroring the seed functions)
    # ----------------------------------------------------------------------- #
    def random_network(self, samples: int) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
        """Generate a new random network with the current architecture."""
        return random_network(self.arch, samples)

    def random_training_data(self, weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
        """Generate training data for a given weight matrix."""
        return random_training_data(weight, samples)

    def fidelity_adjacency(
        self,
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Wrap the original fidelity adjacency construction."""
        return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)

    def state_fidelity(self, a: Tensor, b: Tensor) -> float:
        """Wrap the original state fidelity computation."""
        return state_fidelity(a, b)

    def __repr__(self) -> str:
        return f"<HybridGraphQNN arch={self.arch} head_dim={self.head.out_features} train_circuit={self.train_circuit}>"

# --------------------------------------------------------------------------- #
# Public exports
# --------------------------------------------------------------------------- #

__all__ = [
    "HybridGraphQNN",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
