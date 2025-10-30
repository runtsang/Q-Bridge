import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import itertools
import numpy as np
from typing import List, Tuple, Sequence, Iterable

Tensor = torch.Tensor


def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix of shape (out_features, in_features)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate a dataset that follows the target linear map."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a toy network and associated training data."""
    weights: List[Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight


def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Squared overlap of two normalized vectors."""
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
    for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(s_i, s_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


class GraphQNNGen:
    """
    Hybrid graph‑QNN training loop with a Laplacian regulariser.

    The class trains a simple feed‑forward network that mimics the last
    layer of a quantum graph neural network.  A Laplacian penalty on
    the fidelity adjacency of the layerwise states is added to the loss
    to encourage low‑rank graph structures.
    """

    def __init__(
        self,
        qnn_arch: Sequence[int],
        lr: float = 1e-3,
        epochs: int = 200,
        reg_weight: float = 0.01,
    ):
        self.arch = list(qnn_arch)
        self.lr = lr
        self.epochs = epochs
        self.reg_weight = reg_weight

        layers = []
        for in_f, out_f in zip(self.arch[:-1], self.arch[1:]):
            layers.append(nn.Linear(in_f, out_f))
            layers.append(nn.Tanh())
        self.model = nn.Sequential(*layers)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

    def _laplacian_penalty(self, states: List[Tensor]) -> Tensor:
        adj = fidelity_adjacency(states, threshold=0.8)
        lap = nx.laplacian_matrix(adj).astype(float)
        eigs = torch.from_numpy(np.linalg.eigvalsh(lap.todense()))
        return torch.sum(eigs)

    def train(self, dataset: Iterable[Tuple[Tensor, Tensor]]):
        last_loss = None
        for _ in range(self.epochs):
            for features, target in dataset:
                self.optimizer.zero_grad()
                output = self.model(features)
                loss = self.loss_fn(output, target)

                # Compute activations for regulariser
                activations = [features]
                current = features
                for layer in self.model:
                    current = layer(current)
                    activations.append(current)

                reg = self._laplacian_penalty(activations)
                loss = loss + self.reg_weight * reg

                loss.backward()
                self.optimizer.step()
                last_loss = loss.detach()
        return last_loss

    def run(self, dataset: Iterable[Tuple[Tensor, Tensor]]) -> dict:
        final_loss = self.train(dataset)

        # Estimate target weight via linear least‑squares
        X = torch.stack([f for f, _ in dataset])
        Y = torch.stack([t for _, t in dataset])
        target_weight_est = torch.linalg.lstsq(X, Y).solution

        learned_weight = list(self.model.parameters())[-2].data
        fid = state_fidelity(learned_weight.flatten(), target_weight_est.flatten())

        # Graph diagnostics from final activations
        activations = []
        for features, _ in dataset:
            current = features
            for layer in self.model:
                current = layer(current)
            activations.append(current)

        graph = fidelity_adjacency(activations, threshold=0.8)
        lap = nx.laplacian_matrix(graph).astype(float)
        eigs = np.linalg.eigvalsh(lap.todense())

        return {
            "loss": float(final_loss.item()),
            "fidelity": float(fid),
            "laplacian_spectrum": eigs.tolist(),
        }


__all__ = [
    "GraphQNNGen",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "fidelity_adjacency",
]
