"""GraphQNN__gen149: Classical GNN with quantum‑inspired training hooks.

This module extends the original GraphQNN utilities by adding:
- a hybrid training loop that can operate on both classical tensors and quantum states.
- a fidelity‑based loss function for quantum outputs.
- a decorator that records state trajectories as weighted graphs.
"""

from __future__ import annotations

import itertools
from typing import Iterable, Sequence, List, Tuple, Callable

import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx

Tensor = torch.Tensor

def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a weight matrix initialized from a standard normal distribution."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate a dataset where the target is the linear transformation defined by ``weight``."""
    return [(torch.randn(weight.size(1)), weight @ torch.randn(weight.size(1))) for _ in range(samples)]

def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random linear network together with a target weight for the last layer."""
    weights: List[Tensor] = [_random_linear(in_f, out_f) for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:])]
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight

def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Forward pass that stores all intermediate activations."""
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
    """Squared overlap of two equal‑length tensors after normalisation."""
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
    """Build a weighted graph from pairwise state fidelities."""
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
#  Hybrid training utilities – new additions
# --------------------------------------------------------------------------- #

class FidelityLoss(nn.Module):
    """Loss that penalises 1 - fidelity between output and target."""
    def __init__(self, target: Tensor):
        super().__init__()
        self.target = target.detach()

    def forward(self, output: Tensor) -> Tensor:
        out_norm = output / (torch.norm(output) + 1e-12)
        return 1.0 - (out_norm @ self.target).abs()

def train_loop(
    model: nn.Module,
    data: Iterable[Tuple[Tensor, Tensor]],
    optimizer: optim.Optimizer,
    loss_fn: Callable[[Tensor, Tensor], Tensor] = nn.MSELoss(),
    epochs: int = 200,
    log_interval: int = 20,
) -> List[float]:
    """Standard supervised training loop that records loss values."""
    model.train()
    losses: List[float] = []
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for features, target in data:
            optimizer.zero_grad()
            output = model(features)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if epoch % log_interval == 0:
            losses.append(epoch_loss / len(data))
    return losses

def graph_trace(
    fn: Callable[[Iterable[Tuple[Tensor, Tensor]]], List[List[Tensor]]],
    *,
    threshold: float,
    secondary: float | None = None,
    name: str | None = None,
) -> Callable[[Iterable[Tuple[Tensor, Tensor]]], List[List[Tensor]]]:
    """Decorator that records fidelity‑based graphs of the stored activations."""
    def wrapper(samples: Iterable[Tuple[Tensor, Tensor]]):
        activations = fn(samples)
        flat_states = [act for batch in activations for act in batch]
        if name:
            nx.write_graphml(
                fidelity_adjacency(
                    flat_states,
                    threshold,
                    secondary=secondary,
                ),
                f"{name}.graphml",
            )
        return activations
    return wrapper

__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "train_loop",
    "graph_trace",
    "FidelityLoss",
    "nn",
    "optim",
]
