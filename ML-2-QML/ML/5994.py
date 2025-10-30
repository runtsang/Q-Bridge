"""Hybrid feature‑extractor neural network that learns a data‑dependent map for
the QNN.  The module builds a classical feed‑forward network, trains it with a
regularised loss, and then uses the output as a feature vector for a
parameter‑shift differentiable QNN.
"""

from __future__ import annotations

import itertools
from typing import Iterable, Sequence, List, Tuple, Callable

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
#  Utility helpers
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int, bias: bool = True) -> nn.Linear:
    """Return a fully‑connected layer with random weights."""
    return nn.Linear(in_features, out_features, bias=bias)


def random_training_data(weight: Tensor, samples: int, *, noise: float = 0.0) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic training pairs (x, y) where y = Wx + ε."""
    X = torch.randn(samples, weight.shape[1], dtype=torch.float32)
    y = weight @ X.t()
    y = y.t() + noise * torch.randn_like(y)
    return list(zip(X, y))


def random_network(qnn_arch: Sequence[int], samples: int, *, bias: bool = True) -> Tuple[List[int], List[nn.Module], List[Tuple[Tensor, Tensor]], Tensor]:
    """Create a random classical network and train it to map input to target
    weight matrix.  The output layer has dimensionality equal to the target
    weight matrix.
    """
    # Build layers
    layers: List[nn.Module] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        layers.append(_random_linear(in_f, out_f, bias=bias))
        layers.append(nn.Tanh())
    layers.pop()  # remove trailing activation
    net = nn.Sequential(*layers)

    # Target weight matrix (used as training target)
    target_weight = torch.randn(qnn_arch[-1], qnn_arch[0], dtype=torch.float32)

    # Generate training data
    data = random_training_data(target_weight, samples)

    # Train the network
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    epochs = 200
    for epoch in range(epochs):
        epoch_loss = 0.0
        for x, y in data:
            optimizer.zero_grad()
            out = net(x)
            loss = criterion(out, y) + 1e-4 * torch.norm(out, p=2)  # L2 regularisation
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if epoch % 50 == 0:
            print(f"[ML] Epoch {epoch} loss {epoch_loss / len(data):.4f}")

    return list(qnn_arch), list(net.children()), data, target_weight


def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[nn.Module],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Forward pass through the classical network.  Each sample is a tuple
    (x, y); only the input `x` is forwarded and the intermediate activations
    are collected.
    """
    activations: List[List[Tensor]] = []
    for x, _ in samples:
        current = x
        layerwise: List[Tensor] = [current]
        for layer in weights:
            current = layer(current)
            layerwise.append(current)
        activations.append(layerwise)
    return activations


def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Return squared inner product between two classical feature vectors."""
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
    """Create adjacency graph from feature‑vector fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(s_i, s_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


# --------------------------------------------------------------------------- #
#  GraphQNN class
# --------------------------------------------------------------------------- #
class GraphQNN:
    """
    Classical feature‑extractor that learns a mapping from raw inputs to a
    feature space suitable for a variational quantum circuit.  The API mirrors
    the original seed but adds a regularised training loop.
    """

    def __init__(self, arch: Sequence[int], samples: int, bias: bool = True) -> None:
        self.arch = list(arch)
        self.samples = samples
        self.bias = bias
        self.net, self.data, self.target = self._build_and_train()

    def _build_and_train(self) -> Tuple[nn.Sequential, List[Tuple[Tensor, Tensor]], Tensor]:
        # Build layers
        layers: List[nn.Module] = []
        for in_f, out_f in zip(self.arch[:-1], self.arch[1:]):
            layers.append(_random_linear(in_f, out_f, bias=self.bias))
            layers.append(nn.Tanh())
        layers.pop()  # remove trailing activation
        net = nn.Sequential(*layers)

        # Target weight matrix
        target_weight = torch.randn(self.arch[-1], self.arch[0], dtype=torch.float32)

        # Generate training data
        data = random_training_data(target_weight, self.samples)

        # Train the network
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        epochs = 200
        for epoch in range(epochs):
            epoch_loss = 0.0
            for x, y in data:
                optimizer.zero_grad()
                out = net(x)
                loss = criterion(out, y) + 1e-4 * torch.norm(out, p=2)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            if epoch % 50 == 0:
                print(f"[ML] Epoch {epoch} loss {epoch_loss / len(data):.4f}")

        return net, data, target_weight

    def feedforward(self, samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
        """Compute layer‑wise activations for a batch of samples."""
        return feedforward(self.arch, self.net, samples)

    def fidelity_adjacency(self, states: Sequence[Tensor], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
        """Build a weighted graph from feature‑vector fidelities."""
        return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)

    def __repr__(self) -> str:
        return f"GraphQNN(arch={self.arch}, samples={self.samples}, bias={self.bias})"

__all__ = [
    "GraphQNN",
    "random_network",
    "random_training_data",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
