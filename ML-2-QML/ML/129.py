"""GraphQNN__gen142: Classical graph neural network with fidelity graph classifier.

This module extends the original seed by adding a lightweight GCN that learns from
fidelity graphs.  The public API mirrors the seed for compatibility but
introduces the GraphQNN class that bundles network construction,
forward propagation, graph building, and classifier training.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Iterable, Sequence
from typing import List, Optional, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor
log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# 1. Utility functions (adapted from the seed)
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix of shape (out_features, in_features)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic data for a linear target model."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random weight matrix list and synthetic data."""
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
    """Forward‑pass through the network, recording activations."""
    activations_per_sample: List[List[Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        activations_per_sample.append(activations)
    return activations_per_sample

def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Squared overlap between two state vectors."""
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
    """Build a weighted graph from pairwise state fidelities."""
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
# 2. Simple GCN implementation
# --------------------------------------------------------------------------- #
class SimpleGCN(nn.Module):
    """A two‑layer GCN that operates on an adjacency matrix and node features."""
    def __init__(self, in_features: int, hidden_dim: int, out_features: int):
        super().__init__()
        self.lin1 = nn.Linear(in_features, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_features)

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        deg = adj.sum(dim=1)
        deg_inv_sqrt = torch.diag(1.0 / torch.sqrt(deg + 1e-8))
        norm_adj = deg_inv_sqrt @ adj @ deg_inv_sqrt
        h = F.relu(self.lin1(norm_adj @ x))
        return self.lin2(norm_adj @ h)

# --------------------------------------------------------------------------- #
# 3. GraphQNN class
# --------------------------------------------------------------------------- #
class GraphQNN:
    """Unified interface for classical graph‑based neural network experiments."""
    def __init__(
        self,
        qnn_arch: Sequence[int],
        graph_hidden: int = 32,
        num_classes: int = 2,
        device: torch.device | str = "cpu",
    ):
        self.qnn_arch = list(qnn_arch)
        self.device = torch.device(device)
        self.weights: Optional[List[Tensor]] = None
        self.training_data: Optional[List[Tuple[Tensor, Tensor]]] = None
        self.graph_classifier = SimpleGCN(
            in_features=self.qnn_arch[-1],
            hidden_dim=graph_hidden,
            out_features=num_classes,
        ).to(self.device)

    def build_random(self, samples: int = 1000):
        """Generate random weights and synthetic dataset."""
        _, self.weights, self.training_data, _ = random_network(self.qnn_arch, samples)

    def forward(self, samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
        """Run feed‑forward and return layer activations."""
        if self.weights is None:
            raise RuntimeError("Call build_random() before forward().")
        return feedforward(self.qnn_arch, self.weights, samples)

    def build_fidelity_graph(self, state_list: Sequence[Tensor], threshold: float) -> nx.Graph:
        """Construct a graph from state fidelities."""
        return fidelity_adjacency(state_list, threshold)

    def train_classifier(
        self,
        epochs: int = 50,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        verbose: bool = False,
    ) -> None:
        """Train the GCN on the fidelity graph built from the training data."""
        if self.training_data is None or self.weights is None:
            raise RuntimeError("Call build_random() before training.")
        # Prepare node features: use activations of the last layer
        activations = self.forward(self.training_data)
        last_layer_features = torch.stack([acts[-1] for acts in activations]).to(self.device)
        # Build graph from last-layer states
        state_list = [last_layer_features[i].cpu() for i in range(len(last_layer_features))]
        graph = self.build_fidelity_graph(state_list, threshold=0.9)
        adj = nx.to_numpy_array(graph)
        adj = torch.tensor(adj, dtype=torch.float32, device=self.device)
        # Random binary labels for illustration
        labels = torch.randint(0, 2, (len(last_layer_features),), device=self.device)
        optimizer = torch.optim.Adam(self.graph_classifier.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            self.graph_classifier.train()
            optimizer.zero_grad()
            logits = self.graph_classifier(last_layer_features, adj)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            if verbose and (epoch + 1) % 10 == 0:
                log.info(f"Epoch {epoch+1}/{epochs} loss={loss.item():.4f}")

    def predict(self, sample: Tensor) -> torch.Tensor:
        """Predict class for a single sample."""
        if self.weights is None:
            raise RuntimeError("Call build_random() before predict().")
        activations = feedforward(self.qnn_arch, self.weights, [(sample, sample)])
        last_rep = activations[0][-1].to(self.device)
        adj = torch.eye(1, device=self.device)
        logits = self.graph_classifier(last_rep.unsqueeze(0), adj)
        return logits.argmax(dim=1)

__all__ = [
    "GraphQNN",
    "SimpleGCN",
    "random_network",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
