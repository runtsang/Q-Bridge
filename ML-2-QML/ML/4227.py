"""Combined classical Graph‑QNN with self‑attention and regression utilities.

The module re‑implements the original GraphQNN, SelfAttention, and
QuantumRegression seeds, fusing them into a single class
`GraphQNNAttentionRegression`.  All components are fully
trainable with PyTorch.

Usage
-----
>>> from GraphQNN__gen218 import GraphQNNAttentionRegression, RegressionDataset
>>> model = GraphQNNAttentionRegression([4, 8, 4])
>>> dataset = RegressionDataset(samples=200, num_features=4)
>>> loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
>>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
>>> loss_fn = nn.MSELoss()
>>> for epoch in range(10):
...     for batch in loader:
...         loss = model.train_step(optimizer, loss_fn, batch)
...
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn

Tensor = torch.Tensor


# --------------------------------------------------------------------------- #
#  Graph‑QNN helpers (mirroring the original GraphQNN seed)
# --------------------------------------------------------------------------- #

def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic training data from a target weight matrix."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Return architecture, weight list, training data, and target weight."""
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
    """Perform a forward pass through the QNN, returning activations per layer."""
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
    """Squared overlap of two normalized state vectors."""
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
    """Build a weighted adjacency graph from state fidelities."""
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
#  Classical self‑attention helper
# --------------------------------------------------------------------------- #

class ClassicalSelfAttention:
    """Dot‑product self‑attention implemented with PyTorch."""

    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        query = torch.as_tensor(
            inputs @ rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32
        )
        key = torch.as_tensor(
            inputs @ entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32
        )
        value = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()


# --------------------------------------------------------------------------- #
#  Regression utilities (mirroring the original QuantumRegression seed)
# --------------------------------------------------------------------------- #

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data from a superposition of |0> and |1>."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(torch.utils.data.Dataset):
    """Dataset returning state vectors and regression labels."""

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


# --------------------------------------------------------------------------- #
#  GraphQNNAttentionRegression class
# --------------------------------------------------------------------------- #

class GraphQNNAttentionRegression:
    """
    Classical Graph‑QNN + self‑attention + MLP regression head.

    Parameters
    ----------
    qnn_arch : Sequence[int]
        Widths of each QNN layer.
    attention_dim : int, default 4
        Embedding dimension used by the self‑attention block.
    regression_features : int | None, default None
        Feature dimension fed to the regression head.  If ``None`` the
        dimensionality of the last QNN layer is used.
    """

    def __init__(
        self,
        qnn_arch: Sequence[int],
        attention_dim: int = 4,
        regression_features: int | None = None,
    ):
        self.qnn_arch = list(qnn_arch)
        _, self.weights, _, _ = random_network(self.qnn_arch, samples=1)
        self.attention = ClassicalSelfAttention(embed_dim=attention_dim)
        self.regression_features = regression_features or self.qnn_arch[-1]
        self.regression_head = nn.Sequential(
            nn.Linear(self.regression_features, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    # --------------------------------------------------------------------- #
    #  Forward pass
    # --------------------------------------------------------------------- #
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the QNN, self‑attention, and regression head.

        Parameters
        ----------
        features : torch.Tensor
            Input features of shape ``(batch, in_features)``.

        Returns
        -------
        torch.Tensor
            Predicted scalar values of shape ``(batch,)``.
        """
        # QNN feedforward
        activations = [features]
        current = features
        for weight in self.weights:
            current = torch.tanh(weight @ current)
            activations.append(current)

        # Self‑attention on the last activation
        attn_input = activations[-1].detach().cpu().numpy()
        rotation_params = np.random.normal(
            size=(self.attention.embed_dim, self.attention.embed_dim)
        )
        entangle_params = np.random.normal(
            size=(self.attention.embed_dim, self.attention.embed_dim)
        )
        attn_output = self.attention.run(rotation_params, entangle_params, attn_input)

        # Regression head
        attn_tensor = torch.as_tensor(attn_output, dtype=torch.float32)
        preds = self.regression_head(attn_tensor)
        return preds.squeeze(-1)

    # --------------------------------------------------------------------- #
    #  Training utilities
    # --------------------------------------------------------------------- #
    def train_step(
        self,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        batch: dict[str, torch.Tensor],
    ) -> float:
        optimizer.zero_grad()
        preds = self.forward(batch["states"])
        loss = loss_fn(preds, batch["target"])
        loss.backward()
        optimizer.step()
        return loss.item()

    # --------------------------------------------------------------------- #
    #  Evaluation utilities
    # --------------------------------------------------------------------- #
    def evaluate(self, data_loader: Iterable[dict[str, torch.Tensor]]) -> float:
        total = 0.0
        count = 0
        with torch.no_grad():
            for batch in data_loader:
                preds = self.forward(batch["states"])
                total += ((preds - batch["target"]) ** 2).sum().item()
                count += batch["states"].size(0)
        return total / count

    # --------------------------------------------------------------------- #
    #  Graph utilities
    # --------------------------------------------------------------------- #
    def activation_graph(self, samples: Iterable[Tuple[Tensor, Tensor]]) -> nx.Graph:
        activations = feedforward(self.qnn_arch, self.weights, samples)
        last_states = [act[-1] for act in activations]
        return fidelity_adjacency(last_states, threshold=0.9)


__all__ = [
    "GraphQNNAttentionRegression",
    "random_network",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
    "generate_superposition_data",
    "RegressionDataset",
]
