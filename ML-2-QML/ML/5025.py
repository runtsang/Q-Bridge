"""Hybrid classical graph neural network module.

This module merges classical GNN utilities with quantum‑friendly extensions
such as self‑attention, fully‑connected quantum layer emulation and a
regression dataset that matches the quantum version.  All functions are
purely classical and can be used as drop‑in replacements for the
original GraphQNN, FCL, SelfAttention and QuantumRegression
implementations.
"""

from __future__ import annotations

import itertools
from typing import Iterable, Sequence, List, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import networkx as nx

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
#  Classical utilities – largely retained from the original GraphQNN
# --------------------------------------------------------------------------- #

def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix matching PyTorch's default layout."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate a synthetic regression dataset for a linear target."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    """Return a random feed‑forward architecture together with a dataset."""
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
    """Return the activations for each layer of a classical network."""
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
    """Build a weighted graph from pairwise fidelities."""
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
#  Quantum‑friendly helpers – emulate the QML API in a classical context
# --------------------------------------------------------------------------- #

class SelfAttention:
    """Classic implementation of a self‑attention block."""
    def __init__(self, embed_dim: int = 4):
        self.embed_dim = embed_dim

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        query = torch.as_tensor(inputs @ rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        key   = torch.as_tensor(inputs @ entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        value = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()

class FCL:
    """Simple classical surrogate for the quantum fully‑connected layer."""
    def __init__(self, n_features: int = 1):
        self.linear = nn.Linear(n_features, 1)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.linear(values)).mean(dim=0)
        return expectation.detach().numpy()

# --------------------------------------------------------------------------- #
#  Regression dataset and model – mirrored from the quantum version
# --------------------------------------------------------------------------- #

def generate_superposition_data(num_features: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate data for a simple trigonometric regression task."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(torch.utils.data.Dataset):
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {"states": torch.tensor(self.features[index], dtype=torch.float32),
                "target": torch.tensor(self.labels[index], dtype=torch.float32)}

class QModel(nn.Module):
    """A purely classical regression network that mirrors the quantum model."""
    def __init__(self, num_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(state_batch.to(torch.float32)).squeeze(-1)

# --------------------------------------------------------------------------- #
#  Hybrid network builder – assemble classical & quantum layers
# --------------------------------------------------------------------------- #

class HybridGraphQNN:
    """A hybrid architecture that can intermix classical and quantum layers.

    Parameters
    ----------
    layer_specs : Sequence[Dict[str, Any]]
        Each entry must contain ``type`` (``'classical'`` or ``'quantum'``)
        and layer‑specific arguments.  For quantum layers the module
        must expose a callable ``run`` that accepts a state and returns
        the next state.
    """
    def __init__(self, layer_specs: Sequence[Dict[str, Any]]):
        self.layers = []
        for spec in layer_specs:
            if spec["type"] == "classical":
                self.layers.append(nn.Linear(*spec["shape"]))
            elif spec["type"] == "quantum":
                # The quantum layer is provided by the QML module; we only
                # keep a placeholder reference that will be swapped in at runtime.
                self.layers.append(spec["layer"])
            else:
                raise ValueError(f"Unknown layer type {spec['type']}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.layers:
            if isinstance(layer, nn.Module):
                out = layer(out)
            else:
                # Assume a quantum layer with a `run` method accepting a torch tensor
                out = torch.tensor(layer.run(out.numpy()), dtype=out.dtype, device=out.device)
        return out

__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "SelfAttention",
    "FCL",
    "RegressionDataset",
    "QModel",
    "HybridGraphQNN",
]
