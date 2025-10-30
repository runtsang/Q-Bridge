"""GraphQNN__gen189: Classical graph neural network with differentiable training.

This module extends the original seed by adding:
* Learnable weight parameters using torch.nn.Parameter.
* An Adam optimizer and a simple MSE loss.
* A training loop that iterates over synthetic data.
* A method to evaluate the learned network by constructing a fidelity graph.

The public API mirrors the seed functions but is now fully compatible with PyTorch autograd.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import torch
import torch.nn.functional as F

Tensor = torch.Tensor


def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random matrix of shape (out_features, in_features)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate (x, y) pairs where y = W · x."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        x = torch.randn(weight.size(1), dtype=torch.float32)
        y = weight @ x
        dataset.append((x, y))
    return dataset


def random_network(
    qnn_arch: Sequence[int], samples: int
) -> Tuple[List[int], List[torch.nn.Parameter], List[Tuple[Tensor, Tensor]], Tensor]:
    """
    Create a fully‑connected network with learnable weights.

    The last layer is **not** learnable – it is the ground‑truth target
    for training.  All other layers are wrapped in `torch.nn.Parameter`
    to allow autograd.
    """
    weights: List[torch.nn.Parameter] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:-1]):
        weights.append(torch.nn.Parameter(_random_linear(in_f, out_f)))
    # Target weight (non‑learnable): maps input to output directly
    target = _random_linear(qnn_arch[0], qnn_arch[-1])
    training_data = random_training_data(target, samples)
    return list(qnn_arch), weights, training_data, target


def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[torch.nn.Parameter],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Run the network forward and collect activations for every layer."""
    outputs: List[List[Tensor]] = []
    for x, _ in samples:
        activations = [x]
        cur = x
        for w in weights:
            cur = torch.tanh(w @ cur)
            activations.append(cur)
        outputs.append(activations)
    return outputs


def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Return the squared overlap of two vectors (classical analog)."""
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
    """Build a graph from state‑to‑state fidelity."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


class GraphQNN__gen189:
    """A wrapper that provides training and evaluation for the classical network."""

    def __init__(self, qnn_arch: Sequence[int], samples: int = 100, lr: float = 0.01):
        self.arch, self.weights, self.training_data, self.target = random_network(
            qnn_arch, samples
        )
        self.optimizer = torch.optim.Adam(self.weights, lr=lr)

    def train(self, epochs: int = 200, loss_fn=F.mse_loss) -> None:
        """Simple training loop using Adam and MSE loss."""
        for epoch in range(epochs):
            total_loss = 0.0
            for x, y in self.training_data:
                self.optimizer.zero_grad()
                activations = feedforward(self.arch, self.weights, [(x, y)])
                out = activations[0][-1]
                loss = loss_fn(out, y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            if epoch % 20 == 0 or epoch == epochs - 1:
                avg = total_loss / len(self.training_data)
                print(f"Epoch {epoch:3d}/{epochs:3d}  loss={avg:.4f}")

    def evaluate(
        self,
        threshold: float = 0.9,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Return a fidelity‑based adjacency graph of the final layer outputs."""
        outputs = [
            feedforward(self.arch, self.weights, [(x, y)])[0][-1]
            for x, y in self.training_data
        ]
        return fidelity_adjacency(outputs, threshold, secondary=secondary, secondary_weight=secondary_weight)

    def get_arch(self) -> List[int]:
        return list(self.arch)

    def get_weights(self) -> List[torch.nn.Parameter]:
        return self.weights

    def get_target(self) -> Tensor:
        return self.target


__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "GraphQNN__gen189",
]
