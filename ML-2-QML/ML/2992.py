"""Hybrid graph‑quantum neural network implemented in pure PyTorch.

This module mirrors the original GraphQNN while adding a convolutional
encoder and a fidelity‑based graph construction.  It can be used as a
drop‑in replacement for the classical GraphQNN and serves as a baseline
for the quantum variant.

Key features
------------
* Convolutional encoder (resembles the QFCModel in Quantum‑NAT)
* Graph‑structured feed‑forward propagation with tanh activations
* Fidelity‑based adjacency graph creation
* Random network generator producing a sequence of weight matrices
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import torch
import torch.nn as nn
import networkx as nx

Tensor = torch.Tensor


def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix with shape (out_features, in_features)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic feature–target pairs for a given weight matrix."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
    """Generate a random classical network and a training set for its last layer."""
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
    """Forward propagate a batch of samples through the network."""
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
    """Return the squared overlap of two unit‑norm vectors."""
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
    """Construct a weighted graph based on state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


class HybridGraphQNN(nn.Module):
    """Hybrid classical graph‑quantum neural network.

    Parameters
    ----------
    arch : Sequence[int]
        Layer sizes, including input and output dimensions.
    conv_cfg : dict | None
        Configuration for the convolutional encoder.  Keys are
        ``in_channels``, ``out_channels``, ``kernel_size``, ``stride``,
        and ``padding``.  Defaults to a 2‑layer encoder used in
        Quantum‑NAT.
    threshold : float
        Fidelity threshold for adjacency graph construction.
    secondary : float | None
        Optional secondary threshold; edges with fidelity between
        ``secondary`` and ``threshold`` receive ``secondary_weight``.
    secondary_weight : float
        Weight assigned to secondary edges.
    """

    def __init__(
        self,
        arch: Sequence[int],
        conv_cfg: dict | None = None,
        threshold: float = 0.8,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.arch = list(arch)
        self.threshold = threshold
        self.secondary = secondary
        self.secondary_weight = secondary_weight

        # Convolutional encoder
        cfg = conv_cfg or {
            "in_channels": 1,
            "out_channels": 8,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
        }
        self.encoder = nn.Sequential(
            nn.Conv2d(
                cfg["in_channels"],
                cfg["out_channels"],
                kernel_size=cfg["kernel_size"],
                stride=cfg["stride"],
                padding=cfg["padding"],
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(
                cfg["out_channels"],
                cfg["out_channels"] * 2,
                kernel_size=cfg["kernel_size"],
                stride=cfg["stride"],
                padding=cfg["padding"],
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Infer flattened feature dimension
        with torch.no_grad():
            dummy = torch.zeros(1, cfg["in_channels"], 28, 28)  # 28×28 image
            enc_out = self.encoder(dummy)
            flat_dim = enc_out.view(1, -1).size(1)

        # Linear layers
        layers = []
        in_dim = flat_dim
        for out_dim in arch:
            layers.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim
        self.layers = nn.ModuleList(layers)

        # Normalisation
        self.norm = nn.BatchNorm1d(self.arch[-1])

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the encoder, linear layers and normalisation."""
        bsz = x.shape[0]
        enc = self.encoder(x)
        flat = enc.view(bsz, -1)
        out = flat
        for layer in self.layers:
            out = torch.tanh(layer(out))
        return self.norm(out)

    def random_network(self, samples: int) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
        """Generate a random network and training data for the final layer."""
        return random_network(self.arch, samples)

    def feedforward(
        self,
        weights: Sequence[Tensor],
        samples: Iterable[Tuple[Tensor, Tensor]],
    ) -> List[List[Tensor]]:
        """Feedforward using externally supplied weights (for benchmarking)."""
        return feedforward(self.arch, weights, samples)

    def fidelity_adjacency(self, states: Sequence[Tensor]) -> nx.Graph:
        """Build adjacency graph from a list of state vectors."""
        return fidelity_adjacency(
            states,
            self.threshold,
            secondary=self.secondary,
            secondary_weight=self.secondary_weight,
        )

    def random_training_data(self, samples: int) -> List[Tuple[Tensor, Tensor]]:
        """Generate random training data for the final layer."""
        _, _, dataset, _ = self.random_network(samples)
        return dataset


__all__ = ["HybridGraphQNN"]
