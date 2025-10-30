"""Graph‑based quantum neural network classifier (classical backend)."""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple, Optional

import torch
import torch.nn as nn
import networkx as nx

Tensor = torch.Tensor

# ----------------------------------------------------------------------
# Helper utilities (adapted from original GraphQNN)
# ----------------------------------------------------------------------
def _init_random_weights(in_features: int, out_features: int) -> Tensor:
    """Random weight matrix for a linear layer."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def _create_training_set(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic data (x, Wx)."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        x = torch.randn(weight.size(1), dtype=torch.float32)
        y = weight @ x
        dataset.append((x, y))
    return dataset

def _build_random_network(qnn_arch: Sequence[int], samples: int):
    """Return architecture, weights, training data, and target weight."""
    weights: List[Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_init_random_weights(in_f, out_f))
    target = weights[-1]
    training = _create_training_set(target, samples)
    return list(qnn_arch), weights, training, target

def _forward_pass(qnn_arch: Sequence[int], weights: Sequence[Tensor], samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
    """Layer‑wise activations for each sample."""
    activations: List[List[Tensor]] = []
    for x, _ in samples:
        layer_vals = [x]
        current = x
        for w in weights:
            current = torch.tanh(w @ current)
            layer_vals.append(current)
        activations.append(layer_vals)
    return activations

def _state_fidelity(a: Tensor, b: Tensor) -> float:
    """Squared overlap of two normalised vectors."""
    a_n = a / (torch.norm(a) + 1e-12)
    b_n = b / (torch.norm(b) + 1e-12)
    return float((a_n @ b_n).item() ** 2)

def _fidelity_graph(states: Sequence[Tensor], threshold: float, *, secondary: Optional[float] = None, secondary_weight: float = 0.5) -> nx.Graph:
    """Weighted graph from state fidelities."""
    G = nx.Graph()
    G.add_nodes_from(range(len(states)))
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = _state_fidelity(a, b)
        if fid >= threshold:
            G.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            G.add_edge(i, j, weight=secondary_weight)
    return G

# ----------------------------------------------------------------------
# Classical classifier builder (adapted)
# ----------------------------------------------------------------------
def _build_classifier_network(num_features: int, depth: int) -> Tuple[nn.Module, List[int], List[int], List[int]]:
    """Return a feed‑forward network and metadata."""
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: List[int] = []
    for _ in range(depth):
        lin = nn.Linear(in_dim, num_features)
        layers.extend([lin, nn.LeakyReLU()])  # LeakyReLU instead of ReLU
        weight_sizes.append(lin.weight.numel() + lin.bias.numel())
        in_dim = num_features
    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())
    network = nn.Sequential(*layers)
    observables = list(range(2))
    return network, encoding, weight_sizes, observables

# ----------------------------------------------------------------------
# Main class
# ----------------------------------------------------------------------
class GraphQNNClassifier:
    """Unified classical graph‑based QNN and node‑level classifier."""
    def __init__(self, qnn_arch: Sequence[int], classifier_depth: int = 2, device: str = 'cpu'):
        self.arch = list(qnn_arch)
        self.device = torch.device(device)
        self.arch, self.weights, self.training_data, self.target_weight = _build_random_network(self.arch, samples=200)
        self.weights = [w.to(self.device) for w in self.weights]
        self.classifier, self.enc, self.w_sizes, self.obs = _build_classifier_network(self.arch[-1], classifier_depth)
        self.classifier.to(self.device)
        self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()

        # Convert training data to classification labels (random for demo)
        self.classifier_data: List[Tuple[Tensor, int]] = []
        for x, _ in self.training_data:
            label = int(torch.sum(x) > 0)
            self.classifier_data.append((x, label))

    # ------------------------------------------------------------------
    # Graph utilities
    # ------------------------------------------------------------------
    def graph_from_states(self, states: Sequence[Tensor], threshold: float, secondary: Optional[float] = None) -> nx.Graph:
        """Build adjacency graph from state fidelities."""
        return _fidelity_graph(states, threshold, secondary=secondary)

    # ------------------------------------------------------------------
    # Forward propagation
    # ------------------------------------------------------------------
    def forward(self, samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
        """Layer‑wise activations for each sample."""
        return _forward_pass(self.arch, self.weights, samples)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train_classifier(self, epochs: int = 50, lr: float = 1e-3):
        """Train the classifier on the last‑layer embeddings."""
        self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr=lr)
        for _ in range(epochs):
            for x, y in self.classifier_data:
                self.optimizer.zero_grad()
                out = self.classifier(x.unsqueeze(0).to(self.device))
                loss = self.criterion(out, torch.tensor([y], dtype=torch.long, device=self.device))
                loss.backward()
                self.optimizer.step()

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, embedding: Tensor) -> int:
        """Return class label for a single node embedding."""
        with torch.no_grad():
            out = self.classifier(embedding.unsqueeze(0).to(self.device))
            return int(out.argmax(dim=1).item())

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def generate_random_samples(self, samples: int) -> List[Tuple[Tensor, Tensor]]:
        """Generate synthetic data for a given number of samples."""
        return _create_training_set(self.target_weight, samples)

__all__ = ["GraphQNNClassifier"]
