"""Hybrid kernel‑regression engine with classical, variational and graph components.

This module implements the *classical* side of the combined architecture.
It provides a trainable RBF kernel, a graph‑adjacency builder based on
state‑fidelity, and a lightweight regression head that can be trained
on the kernel features.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "QuantumKernelRegressionGraph",
    "RBFKernel",
    "KernelMatrix",
    "GraphAdjacency",
    "RegressionHead",
]


class RBFKernel(nn.Module):
    """Trainable radial‑basis function kernel.

    The gamma parameter is exposed as a learnable weight so that a
    one‑step optimisation can adjust the kernel bandwidth.
    """
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute RBF kernel matrix between two sets of vectors.

        Parameters
        ----------
        x : torch.Tensor
            Shape (n_samples_x, n_features)
        y : torch.Tensor
            Shape (n_samples_y, n_features)

        Returns
        -------
        torch.Tensor
            Kernel matrix of shape (n_samples_x, n_samples_y)
        """
        x_norm = (x**2).sum(dim=1, keepdim=True)
        y_norm = (y**2).sum(dim=1, keepdim=True)
        sqdist = x_norm - 2 * x @ y.t() + y_norm.t()
        return torch.exp(-self.gamma * sqdist)


class KernelMatrix:
    """Utility wrapper to compute a kernel Gram matrix on the CPU."""

    def __init__(self, kernel: nn.Module) -> None:
        self.kernel = kernel

    def __call__(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """
        Compute the Gram matrix between two collections of vectors.

        Parameters
        ----------
        a : Sequence[torch.Tensor]
            List of tensors of shape (n_features,)
        b : Sequence[torch.Tensor]
            List of tensors of shape (n_features,)

        Returns
        -------
        np.ndarray
            Gram matrix of shape (len(a), len(b))
        """
        a_stack = torch.stack(a, dim=0)
        b_stack = torch.stack(b, dim=0)
        return self.kernel(a_stack, b_stack).detach().cpu().numpy()


class GraphAdjacency:
    """Build a weighted graph from pairwise state fidelities.

    The fidelity is defined as the squared cosine similarity of two
    normalized feature vectors.  Edges with fidelity above *threshold*
    receive weight 1.0; a secondary threshold can assign a lower
    weight.
    """

    def __init__(self, threshold: float, secondary: float | None = None,
                 secondary_weight: float = 0.5) -> None:
        self.threshold = threshold
        self.secondary = secondary
        self.secondary_weight = secondary_weight

    def __call__(self, states: Sequence[torch.Tensor]) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
            fid = self._state_fidelity(s_i, s_j)
            if fid >= self.threshold:
                graph.add_edge(i, j, weight=1.0)
            elif self.secondary is not None and fid >= self.secondary:
                graph.add_edge(i, j, weight=self.secondary_weight)
        return graph

    @staticmethod
    def _state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float((a_norm @ b_norm) ** 2)


class RegressionHead(nn.Module):
    """A minimal regression head that accepts kernel or graph features.

    The head is a two‑layer MLP with a ReLU nonlinearity and outputs a
    single scalar value.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class QuantumKernelRegressionGraph(nn.Module):
    """
    A hybrid kernel‑regression model that combines a trainable RBF kernel,
    a graph adjacency derived from kernel feature fidelity, and a
    regression head.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input feature vectors.
    gamma : float, optional
        Initial value for the RBF kernel bandwidth.
    """
    def __init__(self, num_features: int, gamma: float = 1.0) -> None:
        super().__init__()
        self.kernel = RBFKernel(gamma)
        self.kernel_matrix = KernelMatrix(self.kernel)
        self.graph_builder = GraphAdjacency(threshold=0.8, secondary=0.5)
        self.head = RegressionHead(input_dim=num_features)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that returns a regression prediction for each
        input vector in X.

        Parameters
        ----------
        X : torch.Tensor
            Input features of shape (batch_size, num_features).

        Returns
        -------
        torch.Tensor
            Predicted scalar values of shape (batch_size,).
        """
        # Compute kernel matrix against itself to obtain feature embeddings
        kernel_feats = self.kernel(X, X)  # (batch, batch)
        # Build graph from kernel features
        graph = self.graph_builder([kernel_feats[i, i] for i in range(kernel_feats.size(0))])
        # For simplicity, use the diagonal kernel values as features
        diag = torch.diagonal(kernel_feats, dim1=0, dim2=1).unsqueeze(-1)
        return self.head(diag)
