"""Classical implementation of a graph neural network with kernel and adjacency utilities.

This module extends the original GraphQNN utilities by adding a hybrid interface
that can be used interchangeably with the quantum counterpart.  The class
provides methods to generate random networks, perform feed‑forward
propagation, build fidelity‑based adjacency graphs, and evaluate
classical RBF kernels between network outputs.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
from torch import Tensor, nn

# Utility functions ------------------------------------------------------------

def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix of shape (out_features, in_features)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def _random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic training data for a linear layer."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

# Classical GraphQNN class ----------------------------------------------

class GraphQNN(nn.Module):
    """Hybrid classical graph neural network.

    Parameters
    ----------
    qnn_arch : Sequence[int]
        Layer sizes of the network.
    gamma : float, optional
        RBF kernel width used in :meth:`kernel_matrix`.  Default 1.0.
    """

    def __init__(self, qnn_arch: Sequence[int], gamma: float = 1.0) -> None:
        super().__init__()
        self.qnn_arch = list(qnn_arch)
        self.gamma = gamma
        self.weights: List[Tensor] = []

    # ------------------------------------------------------------------
    # Network construction helpers
    # ------------------------------------------------------------------
    def random_network(self, samples: int) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
        """Generate a random network with synthetic training data.

        Returns
        -------
        arch, weights, training_data, target_weight
        """
        self.weights = [_random_linear(in_f, out_f) for in_f, out_f in zip(self.qnn_arch[:-1], self.qnn_arch[1:])]
        target_weight = self.weights[-1]
        training_data = _random_training_data(target_weight, samples)
        return self.qnn_arch, self.weights, training_data, target_weight

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def feedforward(
        self,
        samples: Iterable[Tuple[Tensor, Tensor]]
    ) -> List[List[Tensor]]:
        """Compute activations for each layer of each sample.

        Parameters
        ----------
        samples : Iterable[Tuple[Tensor, Tensor]]
            A sequence of (feature, target) pairs.  Only the feature part is used.

        Returns
        -------
        List[List[Tensor]]
            Outer list over samples, inner list over layer activations.
        """
        all_acts: List[List[Tensor]] = []
        for features, _ in samples:
            acts: List[Tensor] = [features]
            current = features
            for w in self.weights:
                current = torch.tanh(w @ current)
                acts.append(current)
            all_acts.append(acts)
        return all_acts

    # ------------------------------------------------------------------
    # Fidelity‑based graph construction
    # ------------------------------------------------------------------
    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        """Return squared cosine similarity between two vectors."""
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float(torch.dot(a_norm, b_norm).item() ** 2)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5
    ) -> nx.Graph:
        """Build a weighted adjacency graph from state fidelities."""
        G = nx.Graph()
        G.add_nodes_from(range(len(states)))
        for (i, ai), (j, aj) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNN.state_fidelity(ai, aj)
            if fid >= threshold:
                G.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                G.add_edge(i, j, weight=secondary_weight)
        return G

    # ------------------------------------------------------------------
    # Classical kernel utilities
    # ------------------------------------------------------------------
    def kernel_matrix(
        self,
        a: Sequence[Tensor],
        b: Sequence[Tensor]
    ) -> np.ndarray:
        """Return the RBF Gram matrix between two lists of tensors."""
        a = torch.stack([x.view(-1) for x in a])
        b = torch.stack([y.view(-1) for y in b])
        diff = a.unsqueeze(1) - b.unsqueeze(0)
        sq_norm = torch.sum(diff * diff, dim=2)
        return torch.exp(-self.gamma * sq_norm).numpy()

# ----------------------------------------------------------------------
# Expose public API
# ----------------------------------------------------------------------
__all__ = [
    "GraphQNN",
]
