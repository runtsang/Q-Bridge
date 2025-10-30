"""Hybrid classical graph neural network with embedded QCNN feature extractor.

The module combines the graph-based feedforward utilities from GraphQNN
with a convolution-inspired fully‑connected block derived from QCNNModel.
It provides random network generation, training data synthesis, and
fidelity‑based adjacency construction, all operating in a purely
classical PyTorch/NetworkX stack.

The architecture is:
   1. Node features are passed through a QCNNModel to produce
      enriched embeddings.
   2. The embeddings are aggregated along a graph structure via
      message passing using a simple mean aggregator.
   3. A final classifier head predicts a scalar output per graph.

The module is deliberately lightweight to serve as a research scaffold
for hybrid quantum/classical experiments.
"""

from __future__ import annotations

import itertools
from typing import Iterable, Sequence, List, Tuple, Dict

import torch
import torch.nn as nn
import networkx as nx

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
# 1. Classical QCNN-inspired block
# --------------------------------------------------------------------------- #

class _QCNNBlock(nn.Module):
    """A lightweight fully‑connected block mimicking the QCNN feature map.

    The block is intentionally simple: a sequence of linear layers
    with tanh activations, inspired by the original QCNNModel.
    """
    def __init__(self, in_features: int, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# --------------------------------------------------------------------------- #
# 2. Graph‑aware hybrid model
# --------------------------------------------------------------------------- #

class HybridGraphQCNN(nn.Module):
    """Hybrid graph neural network that embeds QCNN-style feature
    transformations into a message‑passing framework.

    Parameters
    ----------
    node_dim : int
        Dimensionality of raw node features.
    hidden_dim : int
        Hidden dimension used in the QCNN block and the final classifier.
    """
    def __init__(self, node_dim: int, hidden_dim: int = 16):
        super().__init__()
        self.qcnn_block = _QCNNBlock(node_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, graph: nx.Graph, node_features: Dict[int, Tensor]) -> Tensor:
        """
        Parameters
        ----------
        graph : nx.Graph
            Graph whose nodes carry feature tensors.
        node_features : dict[int, Tensor]
            Mapping from node id to feature tensor of shape (node_dim,).

        Returns
        -------
        Tensor
            Scalar prediction per graph (batch‑size 1).
        """
        # Step 1: embed node features via QCNN block
        embedded = {n: self.qcnn_block(f) for n, f in node_features.items()}

        # Step 2: simple mean message passing (one hop)
        aggregated = torch.zeros_like(next(iter(embedded.values())))
        for n, feat in embedded.items():
            neigh = list(graph.neighbors(n))
            if not neigh:
                continue
            neigh_feats = torch.stack([embedded[m] for m in neigh])
            aggregated += neigh_feats.mean(dim=0)
        aggregated /= graph.number_of_nodes()

        # Step 3: classification
        return torch.sigmoid(self.classifier(aggregated))

# --------------------------------------------------------------------------- #
# 3. Utility functions (random data, fidelity, adjacency)
# --------------------------------------------------------------------------- #

def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Random weight matrix with standard normal entries."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(target: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate samples from a linear target mapping."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(target.size(1), dtype=torch.float32)
        dataset.append((features, target @ features))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random classical network mirroring the QNN architecture.

    Returns
    -------
    arch : list[int]
        Architecture list.
    weights : list[Tensor]
        Weight matrices of each layer.
    training_data : list[tuple[Tensor, Tensor]]
        Synthetic dataset produced from the last layer weight.
    target_weight : Tensor
        The final layer weight used as ground truth.
    """
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
    """Classical feed‑forward propagation through the weight matrices."""
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
    """Squared overlap of two classical feature vectors."""
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
    """Construct a weighted graph from pairwise state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


__all__ = [
    "HybridGraphQCNN",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
