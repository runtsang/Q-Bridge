"""ConvGen108: Classical convolution filter with graph utilities.

This module implements a Torch-based convolution filter that emulates a
quantum quanvolution layer, and provides helper functions for
random network generation, feed‑forward evaluation, and
fidelity‑based graph construction.  It extends the original
`Conv.py` by adding trainable weights, a learnable bias, and a
threshold mechanism while keeping the interface drop‑in.
"""

from __future__ import annotations

import torch
from torch import nn
import numpy as np
from typing import Iterable, Tuple, List

import networkx as nx

Tensor = torch.Tensor


class ConvGen108(nn.Module):
    """2‑D convolutional filter that mimics a quantum quanvolution layer.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the square filter.
    in_channels : int, default 1
        Number of input feature maps.
    out_channels : int, default 1
        Number of output feature maps.
    bias : bool, default True
        Whether to add a learnable bias term.
    threshold : float, default 0.0
        Shift applied after the convolution before the sigmoid.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        in_channels: int = 1,
        out_channels: int = 1,
        bias: bool = True,
        threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, bias=bias
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply convolution → sigmoid → mean pooling over spatial dims.

        Parameters
        ----------
        x : Tensor
            Input of shape (batch, channels, H, W).

        Returns
        -------
        Tensor
            Output of shape (batch, out_channels) after sigmoid and
            spatial mean pooling.
        """
        logits = self.conv(x)
        logits = logits - self.threshold
        activations = torch.sigmoid(logits)
        return activations.mean(dim=(2, 3))


# --------------------------------------------------------------------------- #
# Graph utilities mirroring the GraphQNN functions
# --------------------------------------------------------------------------- #

def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix with shape (out_features, in_features)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """
    Generate synthetic training pairs for a single linear layer.

    Parameters
    ----------
    weight : Tensor
        Target weight matrix.
    samples : int
        Number of samples.

    Returns
    -------
    List[Tuple[Tensor, Tensor]]
        List of (features, target) pairs.
    """
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(
    qnn_arch: List[int], samples: int
) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
    """
    Create a random feed‑forward network of linear layers.

    Parameters
    ----------
    qnn_arch : List[int]
        Architecture list, e.g. [4, 8, 16].
    samples : int
        Number of synthetic training samples for the last layer.

    Returns
    -------
    Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]
        Architecture, list of weight matrices, training dataset, and last weight matrix.
    """
    weights: List[Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return qnn_arch, weights, training_data, target_weight


def feedforward(
    qnn_arch: List[int],
    weights: List[Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """
    Run a simple linear network on a dataset.

    Parameters
    ----------
    qnn_arch : List[int]
        Architecture.
    weights : List[Tensor]
        Weight matrices.
    samples : Iterable[Tuple[Tensor, Tensor]]
        Dataset of inputs and targets.

    Returns
    -------
    List[List[Tensor]]
        Activations per sample per layer.
    """
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
    """Overlap between two state vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)


def fidelity_adjacency(
    states: Iterable[Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """
    Build a weighted graph from state fidelities.

    Parameters
    ----------
    states : Iterable[Tensor]
        Sequence of state tensors.
    threshold : float
        Fidelity threshold for weight 1 edges.
    secondary : float, optional
        Secondary threshold for weighted edges.
    secondary_weight : float, default 0.5
        Weight for secondary edges.

    Returns
    -------
    nx.Graph
        Weighted adjacency graph.
    """
    graph = nx.Graph()
    states = list(states)
    graph.add_nodes_from(range(len(states)))
    for i, state_i in enumerate(states):
        for j in range(i + 1, len(states)):
            fid = state_fidelity(state_i, states[j])
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
    return graph


__all__ = [
    "ConvGen108",
    "random_network",
    "feedforward",
    "fidelity_adjacency",
    "state_fidelity",
]
