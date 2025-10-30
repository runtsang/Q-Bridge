"""GraphQNN module for classical machine learning.

This module extends the original GraphQNN implementation by adding
fully connected quantum layer emulation, regression utilities and
kernel methods, while keeping all components purely classical.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import networkx as nx

Tensor = torch.Tensor


class GraphQNN(nn.Module):
    """
    Classical graph neural network that can also emulate quantum layers
    and provide regression and kernel utilities.
    """

    def __init__(self, architecture: Sequence[int], device: str | torch.device = "cpu"):
        super().__init__()
        self.architecture = list(architecture)
        self.device = torch.device(device)
        self.weights = nn.ParameterList(
            nn.Parameter(torch.randn(out, inp, dtype=torch.float32))
            for inp, out in zip(self.architecture[:-1], self.architecture[1:])
        )

    def forward(self, x: Tensor) -> Tensor:
        out = x
        for w in self.weights:
            out = torch.tanh(w @ out)
        return out

    @staticmethod
    def random_network(arch: Sequence[int], samples: int):
        """Generate a random linear network and synthetic training data."""
        weights = [torch.randn(out, inp) for inp, out in zip(arch[:-1], arch[1:])]
        target_weight = weights[-1]
        data: List[Tuple[Tensor, Tensor]] = []
        for _ in range(samples):
            feat = torch.randn(arch[0])
            tgt = target_weight @ feat
            data.append((feat, tgt))
        return list(arch), weights, data, target_weight

    @staticmethod
    def feedforward(
        arch: Sequence[int],
        weights: Sequence[Tensor],
        samples: Iterable[Tuple[Tensor, Tensor]],
    ) -> List[List[Tensor]]:
        """Compute activations for each sample."""
        activations: List[List[Tensor]] = []
        for feat, _ in samples:
            act = [feat]
            cur = feat
            for w in weights:
                cur = torch.tanh(w @ cur)
                act.append(cur)
            activations.append(act)
        return activations

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        """Squared overlap of two normalized vectors."""
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
        """Build a weighted graph from state fidelities."""
        G = nx.Graph()
        G.add_nodes_from(range(len(states)))
        for (i, si), (j, sj) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNN.state_fidelity(si, sj)
            if fid >= threshold:
                G.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                G.add_edge(i, j, weight=secondary_weight)
        return G

    # ------------------------------------------------------------------
    # Regression utilities
    # ------------------------------------------------------------------
    @staticmethod
    def generate_superposition_data(num_features: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create a synthetic regression dataset based on superposition angles."""
        x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
        angles = x.sum(axis=1)
        y = np.sin(angles) + 0.1 * np.cos(2 * angles)
        return x, y.astype(np.float32)

    class RegressionDataset(torch.utils.data.Dataset):
        """Dataset that yields feature vectors and regression targets."""

        def __init__(self, samples: int, num_features: int):
            self.x, self.y = GraphQNN.generate_superposition_data(num_features, samples)

        def __len__(self) -> int:
            return len(self.x)

        def __getitem__(self, idx: int) -> dict[str, Tensor]:
            return {
                "states": torch.tensor(self.x[idx], dtype=torch.float32),
                "target": torch.tensor(self.y[idx], dtype=torch.float32),
            }

    class QModel(nn.Module):
        """Classical neural network used for regression."""

        def __init__(self, num_features: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(num_features, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            )

        def forward(self, x: Tensor) -> Tensor:
            return self.net(x).squeeze(-1)

    # ------------------------------------------------------------------
    # Kernel utilities
    # ------------------------------------------------------------------
    class KernalAnsatz(nn.Module):
        """Radial basis function kernel as a PyTorch module."""

        def __init__(self, gamma: float = 1.0):
            super().__init__()
            self.gamma = gamma

        def forward(self, x: Tensor, y: Tensor) -> Tensor:
            diff = x - y
            return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

    class Kernel(nn.Module):
        """Wrapper that exposes a single scalar kernel value."""

        def __init__(self, gamma: float = 1.0):
            super().__init__()
            self.ansatz = GraphQNN.KernalAnsatz(gamma)

        def forward(self, x: Tensor, y: Tensor) -> Tensor:
            return self.ansatz(x, y).squeeze()

    @staticmethod
    def kernel_matrix(a: Sequence[Tensor], b: Sequence[Tensor], gamma: float = 1.0) -> np.ndarray:
        kernel = GraphQNN.Kernel(gamma)
        return np.array([[kernel(x, y).item() for y in b] for x in a])


__all__ = [
    "GraphQNN",
]
