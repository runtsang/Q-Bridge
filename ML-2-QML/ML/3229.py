"""UnifiedQuantumGraphLayer – classical implementation.

This module provides a single class that can be used as a drop‑in
replacement for the original FCL or GraphQNN seeds.  It exposes a
linear MLP, a `run` method that mimics the quantum interface, and
graph utilities based on state fidelities.  The class inherits from
`torch.nn.Module` so it can be integrated into any PyTorch model.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import torch
from torch import nn

Tensor = torch.Tensor

__all__ = ["UnifiedQuantumGraphLayer"]


class UnifiedQuantumGraphLayer(nn.Module):
    """Hybrid classical‑quantum graph layer.

    The `run` method operates on a list of parameters and returns a
    NumPy array, matching the quantum seed API.  The `forward` method
    implements the standard PyTorch forward pass for a batch of node
    features.  The class also provides static helpers for fidelity
    computation, adjacency graph construction, random network
    generation, and feed‑forward propagation, mirroring the
    GraphQNN utilities.
    """

    def __init__(self, n_features: int = 1, n_qubits: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
        self.n_qubits = n_qubits

    # ------------------------------------------------------------------
    # Quantum‑style API
    # ------------------------------------------------------------------
    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Return the linear output for a list of parameters.

        Parameters are interpreted as scalar weights for the
        linear layer, passed through a tanh activation and averaged
        across the batch, just like the original FCL seed.
        """
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        output = torch.tanh(self.linear(values)).mean(dim=0)
        return output.detach().numpy()

    # ------------------------------------------------------------------
    # PyTorch forward
    # ------------------------------------------------------------------
    def forward(self, features: Tensor) -> Tensor:
        """Standard forward pass for a batch of node features."""
        return torch.tanh(self.linear(features))

    # ------------------------------------------------------------------
    # Graph utilities
    # ------------------------------------------------------------------
    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        """Squared cosine similarity between two feature vectors."""
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float(torch.dot(a_norm, b_norm).item() ** 2)

    @staticmethod
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
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = UnifiedQuantumGraphLayer.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    # ------------------------------------------------------------------
    # Random data generation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
        """Generate synthetic input–target pairs for a linear model."""
        dataset: List[Tuple[Tensor, Tensor]] = []
        for _ in range(samples):
            features = torch.randn(weight.size(1), dtype=torch.float32)
            target = weight @ features
            dataset.append((features, target))
        return dataset

    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int):
        """Create a random linear network for a given architecture."""
        weights: List[Tensor] = []
        for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
            weights.append(torch.randn(out_f, in_f, dtype=torch.float32))
        target_weight = weights[-1]
        training_data = UnifiedQuantumGraphLayer.random_training_data(target_weight, samples)
        return list(qnn_arch), weights, training_data, target_weight

    # ------------------------------------------------------------------
    # Feed‑forward propagation
    # ------------------------------------------------------------------
    @staticmethod
    def feedforward(
        qnn_arch: Sequence[int],
        weights: Sequence[Tensor],
        samples: Iterable[Tuple[Tensor, Tensor]],
    ) -> List[List[Tensor]]:
        """Return activations for each layer over a dataset."""
        stored: List[List[Tensor]] = []
        for features, _ in samples:
            activations = [features]
            current = features
            for weight in weights:
                current = torch.tanh(weight @ current)
                activations.append(current)
            stored.append(activations)
        return stored
