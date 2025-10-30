"""Hybrid classical graph neural network inspired by GraphQNN and Quantum‑NAT.

The class ``GraphQNNHybrid`` encapsulates a feed‑forward network that operates on
graph‑derived node embeddings.  It reuses the fidelity‑based adjacency graph
construction from GraphQNN and augments the forward pass with a 2‑D CNN
feature extractor borrowed from the Quantum‑NAT CNN.  This hybrid design is
intended for experiments that compare classical graph representations with
their quantum counterparts.

The module is fully NumPy / PyTorch based and can be dropped into any
training pipeline that expects a :class:`torch.nn.Module`.  The adjacency graph
provides a convenient way to analyse state similarity during training.
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

# ----- Utility functions (adapted from the original GraphQNN module) -----


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


# ----- Hybrid graph neural network class -----


class GraphQNNHybrid(nn.Module):
    """
    Classical hybrid graph neural network.

    Parameters
    ----------
    qnn_arch : Sequence[int]
        Layer sizes for the inner feed‑forward network.
    conv_channels : int, optional
        Number of channels for the 2‑D convolutional encoder (default 8).
    conv_kernel : int, optional
        Kernel size for the convolutions (default 3).
    """

    def __init__(
        self,
        qnn_arch: Sequence[int],
        conv_channels: int = 8,
        conv_kernel: int = 3,
    ):
        super().__init__()
        self.qnn_arch = list(qnn_arch)
        self.conv = nn.Sequential(
            nn.Conv2d(1, conv_channels, kernel_size=conv_kernel, padding=conv_kernel // 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(conv_channels, conv_channels * 2, kernel_size=conv_kernel, padding=conv_kernel // 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Compute the flattened size after conv layers
        dummy = torch.zeros(1, 1, 28, 28)
        conv_out = self.conv(dummy)
        flat_size = conv_out.view(1, -1).size(1)
        # Build the inner feed‑forward network
        layer_sizes = [flat_size] + list(qnn_arch[1:])
        self.layers = nn.ModuleList(
            [nn.Linear(in_size, out_size) for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:])]
        )
        self.activation = nn.Tanh()
        self.norm = nn.BatchNorm1d(qnn_arch[-1])

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, 1, 28, 28) representing graph adjacency
            matrices rendered as single‑channel images.

        Returns
        -------
        torch.Tensor
            Normalised output of size (N, qnn_arch[-1]).
        """
        bsz = x.shape[0]
        encoded = self.conv(x)
        flat = encoded.view(bsz, -1)
        out = flat
        for layer in self.layers:
            out = self.activation(layer(out))
        return self.norm(out)

    @staticmethod
    def build_random_graph(
        qnn_arch: Sequence[int], samples: int, threshold: float = 0.8
    ) -> nx.Graph:
        """
        Generate a random network and return its fidelity adjacency graph.
        """
        _, weights, _, target = random_network(qnn_arch, samples)
        states = [w.t() for w in weights]
        return fidelity_adjacency(states, threshold)

    @staticmethod
    def random_training_set(
        qnn_arch: Sequence[int], samples: int
    ) -> List[Tuple[Tensor, Tensor]]:
        """
        Return a synthetic dataset for quick sanity checks.
        """
        _, _, dataset, _ = random_network(qnn_arch, samples)
        return dataset

    def train_on_dataset(
        self,
        dataset: List[Tuple[Tensor, Tensor]],
        lr: float = 1e-3,
        epochs: int = 10,
    ) -> None:
        """
        Very small training loop using Adam and MSE loss.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        for _ in range(epochs):
            for features, target in dataset:
                optimizer.zero_grad()
                pred = self.forward(features)
                loss = loss_fn(pred, target)
                loss.backward()
                optimizer.step()


__all__ = [
    "GraphQNNHybrid",
    "random_network",
    "random_training_data",
    "fidelity_adjacency",
    "state_fidelity",
    "feedforward",
]
