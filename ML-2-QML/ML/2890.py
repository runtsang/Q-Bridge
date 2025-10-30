"""Classical graph neural network with convolutional encoding and fully‑connected layers.

The implementation reuses the fidelity‑based utilities from the
original GraphQNN module while adding a lightweight CNN encoder
to process node features.  The architecture is deliberately
aligned with the QFCModel from the Quantum‑NAT example so that the
classical and quantum halves can be swapped seamlessly."""
from __future__ import annotations

import itertools
from typing import List, Tuple, Sequence, Iterable

import torch
import torch.nn as nn
import networkx as nx

Tensor = torch.Tensor


# --------------------------------------------------------------------------- #
#   Utility functions (fidelity, random network, feed‑forward)
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix of shape (out_features, in_features)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate a synthetic dataset using a linear target function."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Build a list of random weight matrices and a training set."""
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
    """Propagate a batch of samples through the network."""
    stored: List[List[Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        for w in weights:
            current = torch.tanh(w @ current)
            activations.append(current)
        stored.append(activations)
    return stored


def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Return the squared overlap of two state vectors."""
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
    """Create a weighted graph from pairwise state fidelities."""
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
#   Classical GraphQNNHybrid model
# --------------------------------------------------------------------------- #
class GraphQNNHybrid(nn.Module):
    """Classical graph neural network with a 1‑D convolutional encoder.

    The encoder processes node feature vectors sequentially, after which
    a fully‑connected backbone produces the final embedding.  The model
    is deliberately simple to allow easy comparison with its quantum
    counterpart."""
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        conv_channels: int = 8,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, conv_channels, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        conv_out = ((in_features + kernel_size - 1) // 2) * conv_channels
        self.fc = nn.Sequential(
            nn.Linear(conv_out, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, out_features),
        )
        self.norm = nn.BatchNorm1d(out_features)

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape (batch, nodes, in_features).

        Returns
        -------
        torch.Tensor
            Normalised embedding of shape (batch, out_features).
        """
        bsz, nodes, feat = x.shape
        x = x.transpose(1, 2)  # (bsz, feat, nodes)
        enc = self.encoder(x)  # (bsz, conv_channels, nodes//2)
        enc = enc.reshape(bsz, -1)
        out = self.fc(enc)
        return self.norm(out)


__all__ = [
    "GraphQNNHybrid",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
