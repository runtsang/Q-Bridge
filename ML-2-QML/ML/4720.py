"""Hybrid classical regression module with graph‑based fidelity regularisation.

The module is fully classical (NumPy/PyTorch) and re‑uses key ideas from the supplied
reference seeds:

* **QuantumRegression** – data generation and a baseline MLP.
* **EstimatorQNN** – a two‑layer Tanh network.
* **GraphQNN** – fidelity‑based graph construction and adjacency handling.

The architecture is split into three logical blocks:

1. **Feature extractor** – a two‑hidden‑layer MLP with Tanh activations.
2. **Regression head** – a linear output layer.
3. **Graph regulariser** – computes a fidelity‑based adjacency graph from
   the hidden‑state trajectory and adds a penalty to the loss.

The design keeps the API compatible with the original ``QuantumRegression`` dataset
and is ready for end‑to‑end training on CPU or GPU.
"""

from __future__ import annotations

import itertools
import numpy as np
import torch
import torch.nn as nn
import networkx as nx
from torch.utils.data import Dataset

# --------------------------------------------------------------------------- #
# Data generation
# --------------------------------------------------------------------------- #
def generate_superposition_data(
    num_features: int, samples: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic regression data that follows
    ``sin(∑x) + 0.1 * cos(2∑x)`` – the same function used in the original seed.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input vectors.
    samples : int
        Number of data points to generate.

    Returns
    -------
    features, labels : tuple[np.ndarray, np.ndarray]
        ``features`` is of shape ``(samples, num_features)``.
        ``labels`` is a 1‑D array of length ``samples``.
    """
    rng = np.random.default_rng()
    x = rng.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """
    Simple PyTorch ``Dataset`` that returns a dictionary with keys
    ``states`` (input vector) and ``target`` (label).
    """
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

# --------------------------------------------------------------------------- #
# Graph utilities (borrowed from GraphQNN)
# --------------------------------------------------------------------------- #
def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Return the squared inner product of two normalized tensors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float((a_norm @ b_norm).item() ** 2)

def fidelity_adjacency(
    states: list[torch.Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """
    Build a weighted adjacency graph from state fidelities.

    Edges with fidelity ≥ ``threshold`` receive weight 1.0.
    When ``secondary`` is provided, fidelities between ``secondary`` and
    ``threshold`` receive ``secondary_weight``.
    """
    G = nx.Graph()
    G.add_nodes_from(range(len(states)))
    for i, j in itertools.combinations(range(len(states)), 2):
        fid = state_fidelity(states[i], states[j])
        if fid >= threshold:
            G.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            G.add_edge(i, j, weight=secondary_weight)
    return G

# --------------------------------------------------------------------------- #
# Hybrid model
# --------------------------------------------------------------------------- #
class HybridRegression(nn.Module):
    """
    Classical regression model with Tanh hidden layers and a graph‑based
    fidelity regulariser.

    The architecture mirrors EstimatorQNN (two hidden layers, Tanh) and
    integrates fidelity‑based adjacency graph construction from GraphQNN.
    """
    def __init__(self, num_features: int, hidden_sizes: tuple[int, int] = (8, 4)):
        super().__init__()
        layers = []
        in_dim = num_features
        for h in hidden_sizes:
            layers.extend([nn.Linear(in_dim, h), nn.Tanh()])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Forward pass that collects activations from each linear layer.

        Returns
        -------
        pred : torch.Tensor
            Regression output of shape ``(batch,)``.
        activations : list[torch.Tensor]
            List of hidden states (after each Linear layer, before activation).
        """
        activations: list[torch.Tensor] = []
        current = x
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                current = layer(current)
                activations.append(current.detach())
            else:  # activation
                current = layer(current)
        pred = current.squeeze(-1)
        return pred, activations

    def graph_regulariser(
        self,
        activations: list[torch.Tensor],
        threshold: float = 0.95,
        secondary: float | None = None,
    ) -> torch.Tensor:
        """
        Compute a fidelity‑based penalty that encourages similarity between
        consecutive hidden states.

        The penalty is the sum over all edges of the adjacency graph of the
        squared difference between the connected node activations, weighted
        by the edge weight.
        """
        G = fidelity_adjacency(activations, threshold, secondary=secondary)
        loss = 0.0
        for i, j, data in G.edges(data=True):
            diff = activations[i] - activations[j]
            loss += data["weight"] * torch.dot(diff, diff)
        return torch.tensor(loss, device=activations[0].device)

__all__ = ["HybridRegression", "RegressionDataset", "generate_superposition_data"]
