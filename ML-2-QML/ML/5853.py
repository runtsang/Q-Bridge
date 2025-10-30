"""GraphQNN__gen252.py – Classical GNN with auto‑encoding and fidelity‑based graph loss.

The module keeps the original feed‑forward and fidelity helpers but adds:
* A small auto‑encoder that learns a latent graph representation.
* A training loop that optimises a hybrid loss (MSE + fidelity).
* A ``train`` method that returns the latent graph and a training history.
* An ``evaluate`` helper that builds a fidelity graph from a test set.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import List, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
#  Core helpers – unchanged from the original seed
# --------------------------------------------------------------------------- #
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

# --------------------------------------------------------------------------- #
#  Auto‑encoder – simple linear encoder/decoder
# --------------------------------------------------------------------------- #
class AutoEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        latent = torch.tanh(self.encoder(x))
        recon = torch.tanh(self.decoder(latent))
        return latent, recon

# --------------------------------------------------------------------------- #
#  Training routine
# --------------------------------------------------------------------------- #
@dataclass
class TrainResult:
    latent_graph: nx.Graph
    history: List[float]

def train(
    qnn_arch: Sequence[int],
    latent_dim: int,
    training_data: List[Tuple[Tensor, Tensor]],
    epochs: int = 200,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> TrainResult:
    device = device or torch.device("cpu")
    input_dim = qnn_arch[0]
    # Build GNN weights
    weights = [_random_linear(in_f, out_f).to(device) for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:])]
    target_weight = weights[-1]  # kept for consistency but unused

    # Auto‑encoder
    ae = AutoEncoder(input_dim, latent_dim).to(device)
    optimizer = optim.Adam(ae.parameters(), lr=lr)
    mse = nn.MSELoss()

    history: List[float] = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for features, target in training_data:
            features = features.to(device)
            target = target.to(device)

            # GNN forward
            current = features
            for w in weights:
                current = torch.tanh(w @ current)
            gnn_output = current

            # Auto‑encoder forward
            latent, recon = ae(features)

            # Loss components
            loss_mse = mse(recon, features)
            loss_fid = 1.0 - state_fidelity(latent, gnn_output)

            loss = loss_mse + loss_fid
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        history.append(epoch_loss / len(training_data))

    # Collect latent representations
    latent_vectors: List[Tensor] = []
    with torch.no_grad():
        for features, _ in training_data:
            latent, _ = ae(features.to(device))
            latent_vectors.append(latent.cpu())

    latent_graph = fidelity_adjacency(latent_vectors, threshold=0.95)

    return TrainResult(latent_graph=latent_graph, history=history)

# --------------------------------------------------------------------------- #
#  Evaluation helper
# --------------------------------------------------------------------------- #
def evaluate(
    qnn_arch: Sequence[int],
    latent_dim: int,
    test_data: List[Tuple[Tensor, Tensor]],
    threshold: float = 0.95,
) -> nx.Graph:
    """Build a fidelity graph of latent representations on a test set."""
    input_dim = qnn_arch[0]
    # Build GNN weights (fixed)
    weights = [_random_linear(in_f, out_f).to("cpu") for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:])]
    target_weight = weights[-1]  # unused but kept

    # Auto‑encoder with random initialization
    ae = AutoEncoder(input_dim, latent_dim)
    latent_vectors: List[Tensor] = []
    with torch.no_grad():
        for features, _ in test_data:
            latent, _ = ae(features)
            latent_vectors.append(latent)

    return fidelity_adjacency(latent_vectors, threshold=threshold)

__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "train",
    "evaluate",
    "AutoEncoder",
    "TrainResult",
]
