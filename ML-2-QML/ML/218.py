import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn

Tensor = torch.Tensor

class GraphQNNGen241:
    """Hybrid classical GraphQNN class.

    Provides classical feed‑forward, fidelity‑based graph construction, and
    a simple MLP predictor built on the graph embeddings.
    """

    @staticmethod
    def feedforward(
        qnn_arch: Sequence[int],
        weights: Sequence[Tensor],
        samples: Iterable[Tuple[Tensor, Tensor]],
    ) -> List[List[Tensor]]:
        """Classical feed‑forward using tanh activations."""
        stored: List[List[Tensor]] = []
        for features, _ in samples:
            activations = [features]
            current = features
            for weight in weights:
                current = torch.tanh(weight @ current)
                activations.append(current)
            stored.append(activations)
        return stored

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        """Return squared cosine similarity between two vectors."""
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float((a_norm @ b_norm).item() ** 2)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Build weighted adjacency graph from state fidelities."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNNGen241.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    @staticmethod
    def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
        """Generate random data for linear target."""
        dataset: List[Tuple[Tensor, Tensor]] = []
        for _ in range(samples):
            features = torch.randn(weight.size(1), dtype=torch.float32)
            target = weight @ features
            dataset.append((features, target))
        return dataset

    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int):
        """Create random linear layers and training data."""
        weights: List[Tensor] = []
        for in_, out in zip(qnn_arch[:-1], qnn_arch[1:]):
            weights.append(GraphQNNGen241._random_linear(in_, out))
        target_weight = weights[-1]
        training_data = GraphQNNGen241.random_training_data(target_weight, samples)
        return list(qnn_arch), weights, training_data, target_weight

    @staticmethod
    def _random_linear(in_features: int, out_features: int) -> Tensor:
        return torch.randn(out_features, in_features, dtype=torch.float32)

    # ------------------------------------------------------------------
    # New experimental methods
    # ------------------------------------------------------------------

    def train_mlp_on_graph(
        self,
        graph: nx.Graph,
        hidden_dim: int = 32,
        epochs: int = 200,
        lr: float = 1e-3,
    ) -> nn.Module:
        """Train a small MLP on the graph Laplacian embeddings."""
        # Compute graph Laplacian and node embeddings
        L = nx.normalized_laplacian_matrix(graph).astype(np.float32)
        embeddings = torch.from_numpy(np.array(L.todense())).float()

        # Simple MLP
        class MLP(nn.Module):
            def __init__(self, in_dim: int, hidden: int):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_dim, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, 1),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.net(x)

        model = MLP(embeddings.size(1), hidden_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        for _ in range(epochs):
            optimizer.zero_grad()
            out = model(embeddings)
            loss = loss_fn(out, torch.zeros_like(out))
            loss.backward()
            optimizer.step()

        return model

    def graph_based_kernel(self, graph: nx.Graph) -> np.ndarray:
        """Return a kernel matrix from the graph adjacency."""
        adj = nx.to_numpy_array(graph)
        # Use cosine similarity as kernel
        norm = np.linalg.norm(adj, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        return (adj @ adj.T) / (norm @ norm.T)
