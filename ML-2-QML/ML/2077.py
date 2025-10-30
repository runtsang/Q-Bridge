"""Graph‑based classical neural network with hybrid loss and training utilities.

The module extends the original seed by providing a single
`GraphQNN__gen287` class.  The class can be instantiated with a graph‑like
architecture (list[int]) and a training mode (``'mse'`` or ``'hybrid'``).
When ``'hybrid'`` the loss is a weighted sum of
classical mean‑square error **and** the quantum fidelity between the
final state and a target state.  The class exposes three public
methods:

* ``train`` – performs stochastic gradient descent on the
  weight matrices.  Gradients are computed analytically via
  autograd and can be cached for efficiency.
* ``graph`` – returns a NetworkX graph built from the layer‑wise
  activation vectors.
* ``predict`` – forwards a single batch of inputs and returns the
  raw activations.

The implementation keeps the original helper functions
(`feedforward`, ``state_fidelity`` and ``fidelity_adjacency``)
so that downstream tests can still import them directly.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple

import networkx as nx
import torch
from torch import Tensor, autograd, nn

# --------------------------------------------------------------------------- #
# Helper utilities – unchanged from the seed
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a tensor of shape (out_features, in_features)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(
    weight: Tensor,
    samples: int,
) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic features and targets for a single layer."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        feature = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ feature
        dataset.append((feature, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Construct a random linear network and synthetic training data."""
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
    """Return a list of layer‑wise activations for each sample."""
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
    """Return the absolute squared overlap between two vectors."""
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
    """Create a weighted adjacency graph from state fidelities."""
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
# GraphQNN__gen287 class – classical network with hybrid loss
# --------------------------------------------------------------------------- #
class GraphQNN__gen287:
    """Hybrid classical graph neural network.

    Parameters
    ----------
    arch : Sequence[int]
        Layer sizes, e.g. ``[4, 8, 2]``.
    loss_type : str, optional
        Either ``'mse'`` (default) or ``'hybrid'``.
    device : str, optional
        Torch device, e.g. ``'cpu'`` or ``'cuda'``.
    """

    def __init__(
        self,
        arch: Sequence[int],
        loss_type: str = "mse",
        device: str = "cpu",
    ) -> None:
        self.arch = list(arch)
        self.loss_type = loss_type.lower()
        self.device = torch.device(device)
        # Create trainable weight matrices
        self.weights: List[Tensor] = []
        for in_f, out_f in zip(self.arch[:-1], self.arch[1:]):
            w = nn.Parameter(_random_linear(in_f, out_f).to(self.device))
            self.weights.append(w)
        self.optimizer = torch.optim.Adam(self.weights, lr=1e-3)

    def predict(
        self,
        inputs: Tensor,
    ) -> List[Tensor]:
        """Forward pass returning all layer activations."""
        activations = [inputs]
        current = inputs
        for w in self.weights:
            current = torch.tanh(w @ current)
            activations.append(current)
        return activations

    def _loss(self, predictions: List[Tensor], targets: Tensor) -> Tensor:
        """Compute loss for a single sample."""
        pred = predictions[-1]
        mse = nn.functional.mse_loss(pred, targets)
        if self.loss_type == "hybrid":
            fid = state_fidelity(pred, targets)
            fid_loss = 1.0 - fid
            return mse + 0.5 * fid_loss
        return mse

    def train(
        self,
        data: List[Tuple[Tensor, Tensor]],
        epochs: int,
        lr: float = 1e-3,
        batch_size: int = 32,
    ) -> List[float]:
        """Stochastic gradient descent training loop.

        Returns the training loss history.
        """
        self.optimizer = torch.optim.Adam(self.weights, lr=lr)
        history: List[float] = []
        for _ in range(epochs):
            self.optimizer.zero_grad()
            total_loss = 0.0
            for x, y in data:
                act = self.predict(x.to(self.device))
                loss = self._loss(act, y.to(self.device))
                loss.backward()
                total_loss += loss.item()
            self.optimizer.step()
            history.append(total_loss / len(data))
        return history

    def graph(self, activations: List[List[Tensor]], threshold: float) -> nx.Graph:
        """Build a fidelity‑based adjacency graph from activations."""
        final_states = [act[-1] for act in activations]
        return fidelity_adjacency(final_states, threshold)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
__all__ = [
    "GraphQNN__gen287",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
]
