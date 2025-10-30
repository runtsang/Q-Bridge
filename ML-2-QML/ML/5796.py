from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import torch
import torch.nn as nn

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
# 1. Classical graph neural network
# --------------------------------------------------------------------------- #
class GraphGNN(nn.Module):
    """
    Minimal feed‑forward graph neural network with tanh activations.
    Architecture is specified by ``arch``: a sequence of integers
    giving the number of features per layer.
    """
    def __init__(self, arch: Sequence[int]):
        super().__init__()
        self.arch = list(arch)
        self.layers = nn.ModuleList(
            [nn.Linear(in_f, out_f) for in_f, out_f in zip(self.arch[:-1], self.arch[1:])]
        )

    def forward(self, x: Tensor) -> Tensor:
        out = x
        for layer in self.layers:
            out = torch.tanh(layer(out))
        return out

# --------------------------------------------------------------------------- #
# 2. Random data generation
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(
    weight: Tensor,
    samples: int,
) -> List[Tuple[Tensor, Tensor]]:
    """Generate ``samples`` input–output pairs for a linear target."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(
    arch: Sequence[int],
    samples: int,
) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
    """Create random weight matrices and training data for the last layer."""
    weights: List[Tensor] = []
    for in_f, out_f in zip(arch[:-1], arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(arch), weights, training_data, target_weight

# --------------------------------------------------------------------------- #
# 3. Feed‑forward and fidelity utilities
# --------------------------------------------------------------------------- #
def feedforward(
    arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Return the list of activations for each sample."""
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
    """Squared overlap between two vectors."""
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
    """Build a weighted graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

def compute_fidelity_graph(
    model: nn.Module,
    dataset: Iterable[Tuple[Tensor, Tensor]],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Run the model on a dataset and construct a fidelity graph."""
    states = [model(x).detach() for x, _ in dataset]
    return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)

# --------------------------------------------------------------------------- #
# 4. Hybrid utilities – converting weights to unitaries
# --------------------------------------------------------------------------- #
def convert_weights_to_unitaries(weights: Sequence[Tensor]) -> List[torch.Tensor]:
    """
    Embed each weight matrix into a unitary via QR decomposition.
    The resulting unitary has the same shape as the weight matrix.
    """
    unitaries = []
    for w in weights:
        q, _ = torch.linalg.qr(w)
        d = torch.diag(_)
        q = q * d.sign()
        unitaries.append(q)
    return unitaries

# --------------------------------------------------------------------------- #
# 5. Training loop
# --------------------------------------------------------------------------- #
def train_gnn(
    arch: Sequence[int],
    training_data: Iterable[Tuple[Tensor, Tensor]],
    epochs: int = 200,
    lr: float = 0.01,
    verbose: bool = False,
) -> nn.Module:
    """Simple MSE‑based training loop for the GraphGNN."""
    model = GraphGNN(arch)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        total_loss = 0.0
        for x, y in training_data:
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if verbose and (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs} loss {total_loss / len(training_data):.4f}")
    return model

__all__ = [
    "GraphGNN",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "compute_fidelity_graph",
    "convert_weights_to_unitaries",
    "train_gnn",
]
