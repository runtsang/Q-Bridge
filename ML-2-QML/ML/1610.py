"""GraphQNN__gen176: classical graph‑neural‑network helper with batch training and hybrid mode."""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor
State = List[Tuple[Tensor, Tensor]]


def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a weight matrix of shape (out_features, in_features)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def _pairwise_cosine_similarity(matrix: Tensor) -> Tensor:
    """Compute pairwise cosine similarity between rows of a matrix."""
    norms = matrix.norm(dim=1, keepdim=True)
    normalized = matrix / (norms + 1e-12)
    return normalized @ normalized.t()


class GraphQNN__gen176(nn.Module):
    """
    A lightweight MLP wrapped in a helper that can also produce a graph from
    node embeddings and run hybrid inference.
    """

    def __init__(
        self,
        qnn_arch: Sequence[int],
        fidelity_threshold: float = 0.9,
        device: str = "cpu",
    ):
        super().__init__()
        self.arch = list(qnn_arch)
        self.fidelity_threshold = fidelity_threshold
        self.device = torch.device(device)

        # Create linear layers
        self.layers = nn.ModuleList()
        for in_f, out_f in zip(self.arch[:-1], self.arch[1:]):
            self.layers.append(nn.Linear(in_f, out_f))

        self.to(self.device)

    def forward(self, x: Tensor) -> Tensor:
        """Standard feed‑forward with tanh activations."""
        for layer in self.layers[:-1]:
            x = torch.tanh(layer(x))
        x = self.layers[-1](x)
        return x

    def train_batch(
        self,
        training_data: List[Tuple[Tensor, Tensor]],
        epochs: int = 10,
        lr: float = 0.01,
        batch_size: int = 32,
    ) -> None:
        """Simple SGD training loop over minibatches."""
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()

        dataset = training_data
        for epoch in range(epochs):
            for i in range(0, len(dataset), batch_size):
                batch = dataset[i : i + batch_size]
                inputs = torch.stack([b[0] for b in batch]).to(self.device)
                targets = torch.stack([b[1] for b in batch]).to(self.device)

                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

    def to_graph(
        self,
        node_features: Tensor,
        threshold: float | None = None,
    ) -> nx.Graph:
        """
        Build a weighted graph from node embeddings using cosine similarity.
        Edges whose similarity exceeds `threshold` are added with weight 1,
        otherwise 0.5 if `secondary` is provided.
        """
        if threshold is None:
            threshold = self.fidelity_threshold

        sim_matrix = _pairwise_cosine_similarity(node_features)
        graph = nx.Graph()
        graph.add_nodes_from(range(node_features.size(0)))

        for i, j in itertools.combinations(range(node_features.size(0)), 2):
            fid = sim_matrix[i, j].item()
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            else:
                graph.add_edge(i, j, weight=0.5)
        return graph

    def state_fidelity(self, a: Tensor, b: Tensor) -> float:
        """Return the squared cosine similarity between two state vectors."""
        a_norm = a / (a.norm() + 1e-12)
        b_norm = b / (b.norm() + 1e-12)
        return float((a_norm @ b_norm).item() ** 2)


class HybridInference:
    """
    Wrapper that can switch between the classical GraphQNN__gen176
    and a quantum variant when requested.
    """

    def __init__(self, classical_qnn: GraphQNN__gen176, quantum_qnn: "QuantumGraphQNN__gen176" | None = None):
        self.classical = classical_qnn
        self.quantum = quantum_qnn

    def run(self, inputs: Tensor, use_quantum: bool = False) -> Tensor:
        if use_quantum and self.quantum is not None:
            return self.quantum.forward(inputs)
        return self.classical.forward(inputs)


__all__ = [
    "GraphQNN__gen176",
    "HybridInference",
]
