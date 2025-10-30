"""
Hybrid Graph Neural Network (GraphQNN) – Classical module.

This module keeps the original public API (`feedforward`, `fidelity_adjacency`,
`random_network`, `random_training_data`, `state_fidelity`) while adding:
* A lightweight :class:`GraphDataset` that stores the generated data and
  provides a ``torch.utils.data.DataLoader`` interface.
* A new ``train_step`` helper that computes a mean‑squared‑error loss between the
  classical network output and the quantum target state (treated as a vector).
  The helper can be used in a pure‑Python training loop or as a PyTorch
  ``autograd`` custom function.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import List, Tuple

import networkx as nx
import torch
from torch.utils.data import Dataset, DataLoader

Tensor = torch.Tensor


def _random_linear(in_features: int, out_features: int) -> Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    weights: List[Tensor] = []
    for in_features, out_features in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_features, out_features))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight


def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
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
# New additions – dataset & training helper
# --------------------------------------------------------------------------- #
@dataclass
class GraphDataset(Dataset):
    """Simple PyTorch dataset wrapping the seed data."""
    features: List[Tensor]
    targets: List[Tensor]

    def __len__(self) -> int:  # pragma: no cover
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self.features[idx], self.targets[idx]


def train_step(
    arch: Sequence[int],
    weights: List[Tensor],
    batch: List[Tuple[Tensor, Tensor]],
    device: torch.device | None = None,
) -> float:
    """
    Compute a mean‑squared‑error loss between the classical network
    output and the quantum target state.  The function is fully
    differentiable when called inside a PyTorch autograd context
    (weights have ``requires_grad=True``).

    Parameters
    ----------
    * arch : sequence of layer sizes
    * *weights : list of learnable weight tensors
    * *batch : list of (feature, target) pairs from the training set
    * * device : optional device for tensors; if omitted, tensors are left on
      their current device.
    """
    batch_features = torch.stack([f for f, _ in batch], dim=0)
    loss = 0.0
    for f, target in batch:
        out = feedforward(arch, weights, [(f, target)])
        out = out[0][-1]
        loss += torch.nn.functional.mse_loss(out, target, reduction="sum")
    return loss / len(batch)


__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "GraphDataset",
    "train_step",
]
