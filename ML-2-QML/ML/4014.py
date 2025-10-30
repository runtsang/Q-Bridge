"""Graph‑based neural network with classical self‑attention.

The class combines the graph‑QNN from the original reference with a
self‑attention operator on node embeddings.  It can generate random
networks, perform a forward pass, and build an adjacency graph based on
state fidelities, mirroring the quantum implementation below."""
from __future__ import annotations

import itertools
import random
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import torch
import numpy as np

Tensor = torch.Tensor


class GraphQNNAttention:
    """Classical graph QNN augmented with a self‑attention layer."""

    def __init__(self, arch: Sequence[int], attention_dim: int = 4, seed: int | None = None):
        self.arch = list(arch)
        self.attention_dim = attention_dim
        self.rng = random.Random(seed)
        self.weights = self._init_weights()

    def _init_weights(self) -> List[Tensor]:
        weights: List[Tensor] = []
        for in_f, out_f in zip(self.arch[:-1], self.arch[1:]):
            w = torch.randn(out_f, in_f, dtype=torch.float32)
            weights.append(w)
        return weights

    @staticmethod
    def random_network(
        arch: Sequence[int], samples: int, seed: int | None = None
    ) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
        """Generate a random network and training data for the last layer."""
        rng = random.Random(seed)
        weights: List[Tensor] = []
        for in_f, out_f in zip(arch[:-1], arch[1:]):
            weights.append(torch.randn(out_f, in_f, dtype=torch.float32))
        target = weights[-1]
        data: List[Tuple[Tensor, Tensor]] = []
        for _ in range(samples):
            x = torch.randn(target.shape[1], dtype=torch.float32)
            y = target @ x
            data.append((x, y))
        return list(arch), weights, data, target

    @staticmethod
    def self_attention(
        inputs: Tensor, params: Tensor, dim: int
    ) -> Tensor:
        """
        Classical self‑attention applied to a batch of node embeddings.
        `params` has shape (dim*3,) for rotations; same as used in the
        original SelfAttention class.
        """
        rot = params.reshape(dim, -1)
        q = torch.tanh(inputs @ rot[:, :dim])
        k = torch.tanh(inputs @ rot[:, dim : 2 * dim])
        v = inputs
        scores = torch.softmax((q @ k.T) / np.sqrt(dim), dim=-1)
        return scores @ v

    def feedforward(
        self, samples: Iterable[Tuple[Tensor, Tensor]]
    ) -> List[List[Tensor]]:
        """Propagate inputs through attention layer and linear layers."""
        outputs: List[List[Tensor]] = []
        for x, _ in samples:
            att = self.self_attention(
                x, torch.randn(self.attention_dim * 3), self.attention_dim
            )
            activations = [att]
            current = att
            for w in self.weights:
                current = torch.tanh(w @ current)
                activations.append(current)
            outputs.append(activations)
        return outputs

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        a_n = a / (torch.norm(a) + 1e-12)
        b_n = b / (torch.norm(b) + 1e-12)
        return float((a_n @ b_n).item() ** 2)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        g = nx.Graph()
        g.add_nodes_from(range(len(states)))
        for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNNAttention.state_fidelity(a, b)
            if fid >= threshold:
                g.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                g.add_edge(i, j, weight=secondary_weight)
        return g


__all__ = ["GraphQNNAttention"]
