"""GraphQNNGen078: classical graph neural network with sampler support.

This module extends the original GraphQNN utilities by exposing a unified
class interface.  It can generate random feed‑forward networks, produce
synthetic training data, run forward passes, compute state fidelities and
build fidelity‑based graphs.  Additionally, a lightweight sampler network
is bundled for quick experiments.
"""

from __future__ import annotations

import itertools
from typing import Iterable, Sequence, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

Tensor = torch.Tensor


def _rand_linear(in_f: int, out_f: int) -> Tensor:
    """Return a random weight matrix of shape (out_f, in_f)."""
    return torch.randn(out_f, in_f, dtype=torch.float32)


def random_training_data(target: Tensor, n_samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic (x, y) pairs for a linear target."""
    data: List[Tuple[Tensor, Tensor]] = []
    for _ in range(n_samples):
        x = torch.randn(target.size(1), dtype=torch.float32)
        y = target @ x
        data.append((x, y))
    return data


def random_network(arch: Sequence[int], n_samples: int):
    """Create a random feed‑forward network and training set."""
    weights: List[Tensor] = [_rand_linear(in_f, out_f) for in_f, out_f in zip(arch[:-1], arch[1:])]
    target = weights[-1]
    train = random_training_data(target, n_samples)
    return list(arch), weights, train, target


def feedforward(arch: Sequence[int], weights: Sequence[Tensor], data: Iterable[Tuple[Tensor, Tensor]]):
    """Run a forward pass through the network, storing activations."""
    activations: List[List[Tensor]] = []
    for x, _ in data:
        layer_out = [x]
        current = x
        for w in weights:
            current = torch.tanh(w @ current)
            layer_out.append(current)
        activations.append(layer_out)
    return activations


def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Overlap squared between two normalized vectors."""
    a_n = a / (torch.norm(a) + 1e-12)
    b_n = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_n, b_n).item() ** 2)


def fidelity_adjacency(states: Sequence[Tensor], thresh: float,
                       *, secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    """Build a weighted graph from fidelity thresholds."""
    g = nx.Graph()
    g.add_nodes_from(range(len(states)))
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(a, b)
        if fid >= thresh:
            g.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            g.add_edge(i, j, weight=secondary_weight)
    return g


# --- Sampler network ---------------------------------------------------------

class _Sampler(nn.Module):
    """A tiny two‑layer network that outputs a probability vector."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4, bias=False),
            nn.Tanh(),
            nn.Linear(4, 2, bias=False),
        )

    def forward(self, inp: Tensor) -> Tensor:  # type: ignore[override]
        return F.softmax(self.net(inp), dim=-1)


def SamplerQNN() -> nn.Module:
    """Return an instance of the sampler network."""
    return _Sampler()


# --- Unified class -----------------------------------------------------------

class GraphQNNGen078:
    """Classical graph‑based neural network with optional sampler.

    Parameters
    ----------
    arch : Sequence[int]
        Layer sizes, e.g. [2, 4, 2].
    mode : str, optional
        Currently only 'classical' is supported; the class can be extended
        to support hybrid or quantum modes in the future.
    """

    def __init__(self, arch: Sequence[int], mode: str = "classical") -> None:
        self.arch = list(arch)
        self.mode = mode
        self.weights, self.train_data, self.target = self._build_random()

    def _build_random(self):
        _, wts, train, tgt = random_network(self.arch, 50)
        return wts, train, tgt

    def train(self, epochs: int = 10, lr: float = 0.01) -> None:
        """A trivial training loop that optimises the final weight matrix."""
        opt = torch.optim.Adam([self.weights[-1]], lr=lr)
        for _ in range(epochs):
            for x, y in self.train_data:
                opt.zero_grad()
                out = feedforward(self.arch, self.weights, [(x, y)])[0][-1]
                loss = F.mse_loss(out, y)
                loss.backward()
                opt.step()

    def forward(self, inputs: Tensor) -> List[Tensor]:
        """Run a forward pass and return activations."""
        return feedforward(self.arch, self.weights, [(inputs, None)])[0]

    def fidelity_graph(self, threshold: float, *, secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
        """Return a graph built from state fidelities of the last layer."""
        last_layer = [act[-1] for act in self.forward(self.train_data[0][0])]
        return fidelity_adjacency(last_layer, threshold,
                                  secondary=secondary,
                                  secondary_weight=secondary_weight)

    def sampler(self, inp: Tensor) -> Tensor:
        """Return the sampler output for a given input."""
        return SamplerQNN()(inp)


__all__ = [
    "GraphQNNGen078",
    "SamplerQNN",
    "random_network",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
