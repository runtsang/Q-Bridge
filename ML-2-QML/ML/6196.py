"""
GraphQuantumNeuralNetwork: classical GNN with hybrid quantum-inspired loss.

The module keeps the original forward‑propagation and fidelity utilities but
extends them with:
* A hybrid loss that combines mean‑squared error on the classical output
  with state‑fidelity on the target quantum state vector.
* A simple training loop that optimises the weight matrices.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix of shape (out_features, in_features)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic (features, target) pairs for a single linear layer."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int) -> tuple[list[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
    """Create a random weight chain and a training set for the last layer."""
    weights: List[Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight


def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Standard multilayer feed‑forward with tanh activations."""
    stored: List[List[Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        stored.append(activations)
    return stored


def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Absolute squared overlap between two real state vectors."""
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
    """Create a weighted adjacency graph from state fidelities.

    Edges with fidelity >= ``threshold`` receive weight 1.
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


class GraphQuantumNeuralNetwork(nn.Module):
    """A hybrid classical‑quantum neural network wrapper.

    Parameters
    ----------
    arch : Sequence[int]
        Layer sizes of the network.
    weights : Sequence[Tensor] | None
        Optional list of weight matrices.  If ``None``, random weights are
        generated internally.
    device : str | torch.device
        Device on which to place the tensors.
    """

    def __init__(
        self,
        arch: Sequence[int],
        weights: Sequence[Tensor] | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.arch = list(arch)
        self.device = torch.device(device)

        if weights is None:
            self.weights = nn.ParameterList(
                [nn.Parameter(_random_linear(in_f, out_f)) for in_f, out_f in zip(self.arch[:-1], self.arch[1:])]
            )
        else:
            self.weights = nn.ParameterList(
                [nn.Parameter(w.to(self.device)) for w in weights]
            )

    @staticmethod
    def random_network(arch: Sequence[int], samples: int) -> tuple[list[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
        """Proxy to the module‑level helper."""
        return random_network(arch, samples)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)

    def forward(self, features: Tensor) -> Tensor:
        """Single forward pass returning the final activation."""
        current = features
        for weight in self.weights:
            current = torch.tanh(weight @ current)
        return current

    def feedforward(
        self,
        samples: Iterable[Tuple[Tensor, Tensor]],
    ) -> List[List[Tensor]]:
        """Return the stored activations for a dataset."""
        return feedforward(self.arch, self.weights, samples)

    def hybrid_loss(
        self,
        outputs: Tensor,
        targets: Tensor,
        target_state: Tensor,
    ) -> Tensor:
        """Hybrid loss: MSE + (1 - fidelity)."""
        mse = F.mse_loss(outputs, targets)
        # Treat the output as a pure state vector
        out_norm = outputs / (torch.norm(outputs) + 1e-12)
        tgt_norm = target_state / (torch.norm(target_state) + 1e-12)
        fidelity = torch.dot(out_norm, tgt_norm).pow(2)
        return mse + (1.0 - fidelity)

    def train_step(
        self,
        optimizer: torch.optim.Optimizer,
        batch: List[Tuple[Tensor, Tensor]],
        target_state: Tensor,
    ) -> Tensor:
        """Single optimizer step on a batch with hybrid loss."""
        optimizer.zero_grad()
        features = torch.stack([f for f, _ in batch]).to(self.device)
        targets = torch.stack([t for _, t in batch]).to(self.device)
        outputs = self.forward(features)
        loss = self.hybrid_loss(outputs, targets, target_state.to(self.device))
        loss.backward()
        optimizer.step()
        return loss


__all__ = [
    "GraphQuantumNeuralNetwork",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
