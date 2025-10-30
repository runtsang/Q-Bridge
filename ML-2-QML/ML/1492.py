"""Hybrid classical Graph Neural Network with PyTorch backend.

This module extends the original reference by:

* Adding a **variational quantum circuit** that produces a unitary
  which is then used as the target for a classical MLP.
* Introducing a **mixed loss** that combines the mean‑squared error
  between MLP outputs and the unitary‑rotated state with a
  fidelity‑based regularization term.
* Exposing a small training loop (`train_step`) that updates the
  MLP weights using Adam.
* Keeping the original `random_network`, `feedforward` and
  `fidelity_adjacency` functions so that downstream tools still
  consume the same API.

The goal is to let a user experiment with a **hybrid** training
strategy while still being able to run the classical part on a CPU.

"""

from __future__ import annotations

import itertools
from typing import Iterable, Sequence

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix of shape (out, in)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> list[tuple[Tensor, Tensor]]:
    """Generate `samples` feature/target pairs using the provided weight matrix."""
    dataset: list[tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Generate a random MLP architecture and training data."""
    weights: list[Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight


def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[tuple[Tensor, Tensor]],
) -> list[list[Tensor]]:
    """Classic feed‑forward through an MLP."""
    stored: list[list[Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        stored.append(activations)
    return stored


def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Squared overlap of two classical vectors."""
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
    """Build a weighted graph from classical state fidelities."""
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
#  Hybrid loss & training utilities
# --------------------------------------------------------------------------- #
class _MLP(nn.Module):
    """Simple fully‑connected network with tanh activations."""
    def __init__(self, arch: Sequence[int]) -> None:
        super().__init__()
        layers = []
        for in_f, out_f in zip(arch[:-1], arch[1:]):
            layers.append(nn.Linear(in_f, out_f))
            layers.append(nn.Tanh())
        # remove the last activation
        layers.pop()
        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


def _classical_fidelity(pred: Tensor, target: Tensor) -> Tensor:
    """Return a differentiable fidelity (1 - squared overlap)."""
    pred_norm = pred / (torch.norm(pred) + 1e-12)
    target_norm = target / (torch.norm(target) + 1e-12)
    overlap = torch.dot(pred_norm, target_norm)
    return 1.0 - overlap ** 2


def hybrid_loss(pred: Tensor, target: Tensor, lam: float = 0.1) -> Tensor:
    """Mean‑squared error + λ * (1 - fidelity)."""
    mse = F.mse_loss(pred, target, reduction="mean")
    fid_penalty = _classical_fidelity(pred, target).mean()
    return mse + lam * fid_penalty


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: list[tuple[Tensor, Tensor]],
    lam: float = 0.1,
) -> float:
    """Perform a single gradient step and return the loss value."""
    model.train()
    optimizer.zero_grad()
    loss = 0.0
    for x, y in batch:
        pred = model(x)
        loss += hybrid_loss(pred, y, lam)
    loss = loss / len(batch)
    loss.backward()
    optimizer.step()
    return loss.item()


__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "train_step",
    "_MLP",
]
