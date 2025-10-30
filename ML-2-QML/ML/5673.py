"""Hybrid classical implementation of graph neural networks with optional RBF kernel support.

This module builds on the original GraphQNN utilities while adding a lightweight kernel
module that mirrors the quantum kernel interface.  All tensors are PyTorch objects and
the API is intentionally compatible with the anchor seed so that downstream experiments
can swap between classical and quantum backends by simply importing the appropriate
module.

Key extensions:
* ``GraphQNNHybrid`` encapsulates the network architecture, data generation, and
  graph construction logic.
* The class exposes a ``kernel_matrix`` method that computes an RBF Gram matrix
  using a configurable gamma parameter, matching the signature of the quantum
  kernel module.
* Helper functions remain fully vectorised for efficiency.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import List, Tuple

import networkx as nx
import numpy as np
import torch
from torch import nn

Tensor = torch.Tensor


class GraphQNNHybrid(nn.Module):
    """Classical graph neural network with RBF kernel support.

    Parameters
    ----------
    architecture:
        Sequence of layer widths, e.g. ``[3, 5, 2]``.
    gamma:
        RBF kernel width.  Defaults to ``1.0`` and is forwarded to
        :meth:`kernel_matrix`.
    """

    def __init__(self, architecture: Sequence[int], gamma: float = 1.0) -> None:
        super().__init__()
        self.arch = tuple(architecture)
        self.gamma = gamma
        # initialise a simple linear stack for demonstration
        self.layers = nn.ModuleList(
            [nn.Linear(in_f, out_f) for in_f, out_f in zip(self.arch[:-1], self.arch[1:])]
        )

    # --------------------------------------------------------------------
    # Data generation helpers
    # --------------------------------------------------------------------
    def random_weights(self) -> List[Tensor]:
        """Return a list of random weight matrices matching ``self.arch``."""
        return [torch.randn(out, in_, dtype=torch.float32) for in_, out in zip(self.arch[:-1], self.arch[1:])]

    def random_training_data(self, weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
        """Generate synthetic input‑output pairs for a given linear map."""
        dataset = []
        for _ in range(samples):
            x = torch.randn(weight.size(1), dtype=torch.float32)
            y = weight @ x
            dataset.append((x, y))
        return dataset

    def random_network(self, samples: int) -> Tuple[Sequence[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
        """Convenience wrapper that creates a random network and a training set."""
        weights = self.random_weights()
        target_weight = weights[-1]
        training_data = self.random_training_data(target_weight, samples)
        return self.arch, weights, training_data, target_weight

    # --------------------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------------------
    def feedforward(
        self,
        weights: Sequence[Tensor],
        samples: Sequence[Tuple[Tensor, Tensor]],
    ) -> List[List[Tensor]]:
        """Compute all layer activations for a batch of samples."""
        activations = []
        for x, _ in samples:
            layer_out = [x]
            current = x
            for w in weights:
                current = torch.tanh(w @ current)
                layer_out.append(current)
            activations.append(layer_out)
        return activations

    # --------------------------------------------------------------------
    # Fidelity utilities
    # --------------------------------------------------------------------
    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        """Squared cosine similarity between two vectors."""
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float((a_norm @ b_norm).item() ** 2)

    def fidelity_adjacency(
        self,
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Build a weighted graph from state fidelities."""
        G = nx.Graph()
        G.add_nodes_from(range(len(states)))
        for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
            fid = self.state_fidelity(s_i, s_j)
            if fid >= threshold:
                G.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                G.add_edge(i, j, weight=secondary_weight)
        return G

    # --------------------------------------------------------------------
    # Kernel utilities
    # --------------------------------------------------------------------
    def _rbf_kernel(self, x: Tensor, y: Tensor) -> Tensor:
        """Element‑wise RBF kernel for two 1‑D tensors."""
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff))

    def kernel_matrix(self, a: Sequence[Tensor], b: Sequence[Tensor]) -> np.ndarray:
        """Compute the Gram matrix between two collections of vectors."""
        matrix = torch.stack([torch.stack([self._rbf_kernel(x, y) for y in b]) for x in a])
        return matrix.numpy()

    # --------------------------------------------------------------------
    # Convenience wrappers
    # --------------------------------------------------------------------
    def graph_from_states(
        self,
        activations: List[List[Tensor]],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Create a graph from the final layer activations of a batch."""
        final_states = [layer[-1] for layer in activations]
        return self.fidelity_adjacency(final_states, threshold, secondary=secondary, secondary_weight=secondary_weight)
