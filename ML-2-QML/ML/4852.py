"""Graph-based neural network with classical and quantum kernel extensions.

This module implements a unified GraphQNNGen313 class that can operate in
classical mode (using PyTorch) or quantum mode (using TorchQuantum).  The
class provides utilities for random network generation, state fidelity
graphs, and both RBF and quantum kernel evaluations.  The design follows
the original GraphQNN and QuantumKernelMethod reference pairs while
adding regression dataset support.

Key features:
- Classical feed‑forward network with tanh activations.
- Quantum‑style kernel module (classical RBF kernel) with the same API.
- Fidelity‑based graph construction for state comparison.
- Regression dataset for supervised learning on superposition data.
"""

from __future__ import annotations

import itertools
from typing import Iterable, Sequence, Tuple, List, Optional

import torch
from torch import nn
import numpy as np
import networkx as nx
from torch.utils.data import Dataset

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
#  Classical Graph Neural Network
# --------------------------------------------------------------------------- #
class GraphQNNGen313(nn.Module):
    """Unified graph‑quantum neural network.

    Parameters
    ----------
    arch : Sequence[int]
        Layer sizes, e.g. ``[4, 8, 2]``.
    gamma : float, optional
        RBF kernel width.  Only used when ``self.kernel`` is instantiated.
    """

    def __init__(self, arch: Sequence[int], gamma: float = 1.0):
        super().__init__()
        self.arch = list(arch)
        self.layers: nn.ModuleList[nn.Linear] = nn.ModuleList()
        for in_f, out_f in zip(self.arch[:-1], self.arch[1:]):
            self.layers.append(nn.Linear(in_f, out_f))

        # Classical RBF kernel
        self.kernel = _RBFKernel(gamma)

    # --------------------------------------------------------------------- #
    #  Forward and utilities
    # --------------------------------------------------------------------- #
    def forward(self, x: Tensor) -> Tensor:
        """Return final layer activations."""
        out = x
        for layer in self.layers:
            out = torch.tanh(layer(out))
        return out.squeeze(-1)

    @staticmethod
    def feedforward(
        arch: Sequence[int], weights: Sequence[Tensor], samples: Iterable[Tuple[Tensor, Tensor]]
    ) -> List[List[Tensor]]:
        """Return activations for each sample under a fixed weight set."""
        activations: List[List[Tensor]] = []
        for feature, _ in samples:
            a = [feature]
            current = feature
            for w in weights:
                current = torch.tanh(w @ current)
                a.append(current)
            activations.append(a)
        return activations

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: Optional[float] = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Build a weighted graph from state fidelities."""
        G = nx.Graph()
        G.add_nodes_from(range(len(states)))
        for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
            fid = _state_fidelity(s_i, s_j)
            if fid >= threshold:
                G.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                G.add_edge(i, j, weight=secondary_weight)
        return G

    @staticmethod
    def random_network(arch: Sequence[int], samples: int):
        """Generate a random classical network and training data."""
        weights: List[Tensor] = []
        for in_f, out_f in zip(arch[:-1], arch[1:]):
            weights.append(torch.randn(out_f, in_f, dtype=torch.float32))
        target = weights[-1]
        train_data = _random_training_data(target, samples)
        return list(arch), weights, train_data, target

    # --------------------------------------------------------------------- #
    #  Kernel utilities
    # --------------------------------------------------------------------- #
    def kernel_matrix(self, a: Sequence[Tensor], b: Sequence[Tensor]) -> np.ndarray:
        """Return Gram matrix using the built‑in RBF kernel."""
        return np.array([[self.kernel(x, y) for y in b] for x in a])

    # --------------------------------------------------------------------- #
    #  Regression dataset utilities
    # --------------------------------------------------------------------- #
    @staticmethod
    def generate_superposition_data(num_features: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return features and labels for a synthetic superposition task."""
        x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
        angles = x.sum(axis=1)
        y = np.sin(angles) + 0.1 * np.cos(2 * angles)
        return x, y.astype(np.float32)

def generate_superposition_data(num_features: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    return GraphQNNGen313.generate_superposition_data(num_features, samples)

class RegressionDataset(Dataset):
    """Simple regression dataset based on superposition states."""

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # noqa: D105
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:  # noqa: D105
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

# --------------------------------------------------------------------------- #
#  Helper functions
# --------------------------------------------------------------------------- #
def _random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate training pairs (x, Wx)."""
    data: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        x = torch.randn(weight.size(1), dtype=torch.float32)
        y = weight @ x
        data.append((x, y))
    return data


def _state_fidelity(a: Tensor, b: Tensor) -> float:
    """Squared overlap of two normalized classical vectors."""
    an = a / (torch.norm(a) + 1e-12)
    bn = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(an, bn).item() ** 2)


class _RBFKernel:
    """Simple RBF kernel for tensor inputs."""

    def __init__(self, gamma: float = 1.0):
        self.gamma = gamma

    def __call__(self, x: Tensor, y: Tensor) -> float:
        diff = x - y
        return float(torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True)).item())

# --------------------------------------------------------------------------- #
#  Exports
# --------------------------------------------------------------------------- #
__all__ = [
    "GraphQNNGen313",
    "RegressionDataset",
    "generate_superposition_data",
]
