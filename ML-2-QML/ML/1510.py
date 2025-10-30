import itertools
from typing import List, Tuple, Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

Tensor = torch.Tensor


def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a weight matrix of shape (out, in) with normalised variance."""
    return torch.randn(out_features, in_features, dtype=torch.float32) / (in_features ** 0.5)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate samples (x, y) where y = weight @ x."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        x = torch.randn(weight.shape[1], dtype=torch.float32)
        y = weight @ x
        dataset.append((x, y))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random linear network and training data for its last layer."""
    weights: List[Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight


class ResidualMLP(nn.Module):
    """A small MLP with skip‑connections that mimics the QNN architecture."""

    def __init__(self, arch: Sequence[int]):
        super().__init__()
        self.arch = list(arch)
        self.layers = nn.ModuleList(
            [nn.Linear(in_f, out_f) for in_f, out_f in zip(self.arch[:-1], self.arch[1:])]
        )
        self.residuals = nn.ModuleList(
            [nn.Identity() if i == 0 else nn.Linear(self.arch[i - 1], self.arch[i]) for i in range(1, len(self.arch))]
        )

    def forward(self, x: Tensor) -> Tensor:
        out = x
        for idx, (lin, res) in enumerate(zip(self.layers, self.residuals)):
            out = F.tanh(lin(out)) + res(out)
            if idx < len(self.arch) - 2:
                out = F.tanh(out)
        return out

    def predict(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def train_one_epoch(
        self,
        data: Iterable[Tuple[Tensor, Tensor]],
        lr: float = 1e-3,
    ) -> float:
        """Train for one epoch and return average MSE loss."""
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        total_loss = 0.0
        for x, y in data:
            optimizer.zero_grad()
            pred = self.forward(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(data)


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


__all__ = [
    "ResidualMLP",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
