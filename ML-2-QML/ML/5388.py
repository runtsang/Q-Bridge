from __future__ import annotations

import itertools
from typing import List, Sequence

import torch
import torch.nn as nn
import numpy as np
import networkx as nx


class QuantumNATGen279(nn.Module):
    """
    Classical hybrid model inspired by Quantum‑NAT, EstimatorQNN, GraphQNN and FCL.
    The architecture combines a 2‑D CNN encoder, a quantum‑inspired unitary
    transformation applied to the flattened features, and a fully‑connected
    output layer.  It also offers utilities to build fidelity‑based adjacency
    graphs of intermediate states.
    """

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Classical approximation of a quantum layer
        self.num_features = 16 * 7 * 7
        self.unitary = self._random_unitary(self.num_features)
        self.fc = nn.Sequential(
            nn.Linear(self.num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)

    # ------------------------------------------------------------------
    # Random unitary generator
    # ------------------------------------------------------------------
    def _random_unitary(self, dim: int) -> torch.Tensor:
        """Generate a random real orthogonal matrix and convert to torch."""
        rng = np.random.default_rng()
        mat = rng.standard_normal((dim, dim))
        q, _ = np.linalg.qr(mat)
        return torch.from_numpy(q.astype(np.float32))

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        features = self.encoder(x).view(bsz, -1)
        transformed = torch.matmul(features, self.unitary)
        out = self.fc(transformed)
        return self.norm(out)

    # ------------------------------------------------------------------
    # Graph utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float(torch.dot(a_norm, b_norm).item() ** 2)

    def fidelity_adjacency(
        self,
        states: List[torch.Tensor],
        threshold: float,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
            fid = self._state_fidelity(a, b)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph


__all__ = ["QuantumNATGen279"]
