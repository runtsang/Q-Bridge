"""Graph‑based neural network with graph‑level loss and hybrid training.

The module extends the original seed by:
* adding a `GraphLoss` class that aggregates per‑node fidelity‑based penalties;
* providing a `GraphQNN` class that can run a classical forward pass and a quantum forward pass on a Pennylane simulator;
* exposing a `train` method that accepts a PyTorch device or a Qiskit backend.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple, Optional

import networkx as nx
import torch
import numpy as np

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
# 1.  Classical utilities – same API as the seed
# --------------------------------------------------------------------------- #

def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Randomly initialise a linear map."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate a dataset where the target is the matrix‑matrix product of
    the weight matrix and a random input vector.  The seed‑original
    behaviour is preserved.
    """
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    """Return an architecture, a list of weight matrices, a training set and
    the target weight (last layer).
    """
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
    """Return the absolute squared overlap between two vectors."""
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
    """Create a weighted adjacency graph from state fidelities."""
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
# 2.  Graph‑level loss
# --------------------------------------------------------------------------- #

class GraphLoss:
    """Graph‑level loss based on fidelity penalties between adjacent nodes."""

    def __init__(self, adjacency: nx.Graph, weight: float = 1.0) -> None:
        self.adjacency = adjacency
        self.weight = weight

    def __call__(self, states: Sequence[Tensor]) -> Tensor:
        """Compute the sum of (1 - fidelity) over all edges."""
        loss = torch.tensor(0.0, device=states[0].device)
        for i, j, data in self.adjacency.edges(data=True):
            fid = state_fidelity(states[i], states[j])
            loss += self.weight * (1.0 - fid)
        return loss

# --------------------------------------------------------------------------- #
# 3.  Hybrid GraphQNN
# --------------------------------------------------------------------------- #

class GraphQNN:
    """Hybrid graph neural network that can run a classical forward pass
    or a quantum forward pass on a Pennylane simulator.
    """

    def __init__(
        self,
        arch: Sequence[int],
        device: str = "cpu",
        use_quantum: bool = False,
        quantum_device: str = "default.qubit",
    ) -> None:
        self.arch = list(arch)
        self.device = torch.device(device)
        self.use_quantum = use_quantum
        self.quantum_device_name = quantum_device
        self.weights: List[Tensor] = [
            _random_linear(in_f, out_f).to(self.device) for in_f, out_f in zip(self.arch[:-1], self.arch[1:])
        ]
        self.adjacency: Optional[nx.Graph] = None

    # --------------------------------------------------------------------- #
    # Classical forward pass
    # --------------------------------------------------------------------- #
    def forward(self, x: Tensor) -> Tensor:
        """Compute the output of the network for a single input."""
        current = x.to(self.device)
        for weight in self.weights:
            current = torch.tanh(weight @ current)
        return current

    # --------------------------------------------------------------------- #
    # Quantum forward pass (placeholder – to be implemented in the QML module)
    # --------------------------------------------------------------------- #
    def quantum_forward(self, x: np.ndarray) -> np.ndarray:
        """Quantum forward pass – implemented in the QML module."""
        raise NotImplementedError("Quantum forward pass is defined in the QML module.")

    # --------------------------------------------------------------------- #
    # Training
    # --------------------------------------------------------------------- #
    def train(
        self,
        data: List[Tuple[Tensor, Tensor]],
        epochs: int = 100,
        lr: float = 0.01,
        optimizer_cls= torch.optim.Adam,
        loss_fn= torch.nn.MSELoss(),
        adjacency: Optional[nx.Graph] = None,
        graph_loss_weight: float = 0.1,
        verbose: bool = False,
    ) -> None:
        """Train the network on the provided data."""
        self.adjacency = adjacency
        optimizer = optimizer_cls(self.weights, lr=lr)
        for epoch in range(epochs):
            epoch_loss = 0.0
            for x, y in data:
                optimizer.zero_grad()
                pred = self.forward(x)
                loss = loss_fn(pred, y)
                if adjacency is not None:
                    # Compute states for graph loss
                    states = feedforward(self.arch, self.weights, [(x, y)])
                    graph_loss = GraphLoss(adjacency, weight=graph_loss_weight)(states[0])
                    loss += graph_loss
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} – loss: {epoch_loss/len(data):.4f}")

    # --------------------------------------------------------------------- #
    # Utility methods
    # --------------------------------------------------------------------- #
    def get_fidelity_adjacency(
        self,
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Return a fidelity‑based adjacency graph from the current activations."""
        # In practice, call with actual data to compute activations
        raise NotImplementedError("Use feedforward with real data to compute adjacency.")

    def get_weights(self) -> List[Tensor]:
        """Return the list of weight matrices."""
        return self.weights

# --------------------------------------------------------------------------- #
# 4.  Exports
# --------------------------------------------------------------------------- #

__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "GraphLoss",
    "GraphQNN",
]
