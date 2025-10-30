"""GraphQNNGen207: hybrid convolutional‑graph neural network.

This module fuses the CNN encoder from Quantum‑NAT with a graph‑based
network that constructs a fidelity‑based adjacency graph from the
activations of the last hidden layer.  It is a classical implementation
that mirrors the interface of the original GraphQNN seed and is fully
PyTorch‑compatible.
"""

import itertools
import math
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import torch
import torch.nn as nn

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
# Utility helpers – same behaviour as the seed module, but adapted to the
# new architecture
# --------------------------------------------------------------------------- #

def _rand_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix of shape (out, in)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate a synthetic training set that targets the given weight."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    """Return a random graph‑based network together with data for
    supervised training."""
    weights: List[Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_rand_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight

def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Return the activations for every layer of the network."""
    outputs: List[List[Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        for w in weights:
            current = torch.tanh(w @ current)
            activations.append(current)
        outputs.append(activations)
    return outputs

def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Cosine similarity squared, safe from zero‑norm."""
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
    """Build a weighted adjacency graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(a, b)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

# --------------------------------------------------------------------------- #
# GraphQNNGen207 – the main hybrid model
# --------------------------------------------------------------------------- #

class GraphQNNGen207(nn.Module):
    """Hybrid convolutional‑graph neural network.

    Parameters
    ----------
    conv_channels : Sequence[int]
        Number of channels for each Conv2d block.
    qnn_arch : Sequence[int]
        Architecture of the graph‑based neural network (input size,
        hidden layers, output size).
    threshold : float
        Fidelity threshold for constructing the adjacency graph.
    secondary : float | None
        Optional secondary threshold.
    secondary_weight : float
        Weight assigned to edges that fall between the two thresholds.
    """

    def __init__(
        self,
        conv_channels: Sequence[int] = (8, 16),
        qnn_arch: Sequence[int] = (64, 32, 16),
        threshold: float = 0.8,
        secondary: float | None = 0.6,
        secondary_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.secondary = secondary
        self.secondary_weight = secondary_weight

        # Convolutional backbone – identical to the QFCModel encoder
        conv_layers = []
        in_ch = 1
        for out_ch in conv_channels:
            conv_layers.extend(
                [
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                ]
            )
            in_ch = out_ch
        self.features = nn.Sequential(*conv_layers)

        # Graph‑based layers
        self.qnn_arch = list(qnn_arch)
        self.weight_layers = nn.ModuleList(
            [nn.Linear(in_f, out_f) for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:])]
        )
        self.norm = nn.BatchNorm1d(qnn_arch[-1])

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Returns the normalized output of the graph network and
        the adjacency graph built from the layer activations.
        """
        bsz = x.shape[0]
        feat = self.features(x)
        flat = feat.view(bsz, -1)

        activations: List[Tensor] = [flat]
        curr = flat
        for layer in self.weight_layers:
            curr = torch.tanh(layer(curr))
            activations.append(curr)

        # Build adjacency graph on the activations of the last hidden layer
        last_state = activations[-1]
        graph = fidelity_adjacency(
            [last_state[i] for i in range(bsz)],
            self.threshold,
            secondary=self.secondary,
            secondary_weight=self.secondary_weight,
        )

        out = self.norm(activations[-1])
        return out, graph

    # ------------------------------------------------------------------ #
    # Convenience helpers – mirror the seed API
    # ------------------------------------------------------------------ #
    def random_network(self, samples: int = 100):
        return random_network(self.qnn_arch, samples)

    def feedforward(self, samples: Iterable[Tuple[Tensor, Tensor]]):
        return feedforward(self.qnn_arch, [l.weight for l in self.weight_layers], samples)

    def fidelity_adjacency(self, states: Sequence[Tensor], threshold: float):
        return fidelity_adjacency(states, threshold)

__all__ = [
    "GraphQNNGen207",
    "random_network",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
