"""GraphQNN__gen301 – Classical GNN with hybrid loss and adjacency regularisation.

The module keeps the original feed‑forward and fidelity helpers but adds:
* A `LayerSelector` that picks a subset of layers to train, enabling curriculum learning.
* A `HybridLoss` that computes MSE on the final layer and adds a fidelity term between the
  output state and a target unitary state.
* A `train_batch` helper that runs a single optimisation step on a mini‑batch.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import torch
import torch.nn.functional as F
from torch import nn

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
# New hybrid features
# --------------------------------------------------------------------------- #

class LayerSelector:
    """Selects a subset of layers to train, optionally based on a curriculum schedule."""
    def __init__(self, total_layers: int, schedule: Sequence[int] | None = None):
        self.total = total_layers
        self.schedule = schedule or list(range(total_layers))
        self.current = 0

    def current_layers(self) -> List[int]:
        """Return indices of the current training set."""
        return list(range(self.current, min(self.current + self.schedule[self.current], self.total)))

    def step(self) -> None:
        """Advance to the next curriculum stage."""
        self.current += 1


class HybridLoss:
    """Hybrid loss combining MSE on the final layer and a fidelity term."""
    def __init__(self, weight: float = 0.5, device: str = "cpu"):
        """weight: 0‑1 weight for the loss‑fidelity term."""
        self.weight = weight
        self.device = device

    def __call__(self, output: Tensor, target: Tensor, target_state: Tensor) -> float:
        mse = F.mse_loss(output, target).item()
        fid = state_fidelity(output, target_state)
        return mse + self.weight * (1 - fid)


def train_batch(
    model: nn.Module,
    weights: List[Tensor],
    batch: List[Tuple[Tensor, Tensor]],
    optimizer: torch.optim.Optimizer,
    loss_fn: HybridLoss,
    device: str = "cpu",
    trainable_layers: List[int] = None,
) -> float:
    """Single optimisation step on a mini‑batch."""
    model.train()
    optimizer.zero_grad()
    activations = feedforward(model.qnn_arch, weights, batch)
    # Only compute gradients for selected layers
    selected = trainable_layers or list(range(len(weights)))
    for idx in selected:
        weights[idx].requires_grad = True
    loss = loss_fn(activations[-1][-1], activations[-1][-1], torch.tensor(0.0, device=device))
    loss.backward()
    optimizer.step()
    return loss.item()


__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "LayerSelector",
    "HybridLoss",
    "train_batch",
]
