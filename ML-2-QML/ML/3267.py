"""GraphQNNGen207: Classical graph‑neural‑network with optional self‑attention.

The class mirrors the original GraphQNN API while adding a Self‑Attention layer that can be toggled per instance.  All tensors are PyTorch objects and graph utilities are from networkx.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the classical self‑attention helper.
from SelfAttention import SelfAttention

Tensor = torch.Tensor


class GraphQNNGen207:
    """
    Classical graph‑neural‑network with optional self‑attention.
    """

    def __init__(self, arch: Sequence[int], use_attention: bool = False):
        """
        Parameters
        ----------
        arch : Sequence[int]
            Layer sizes (including input and output dimensions).
        use_attention : bool, optional
            If True, a SelfAttention block is inserted after each linear layer.
        """
        self.arch = list(arch)
        self.use_attention = use_attention
        self.layers = nn.ModuleList()
        for in_f, out_f in zip(arch[:-1], arch[1:]):
            linear = nn.Linear(in_f, out_f, bias=True)
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            self.layers.append(linear)

        if use_attention:
            # SelfAttention returns a class; instantiate it with default embed_dim.
            self.attention = SelfAttention()

    @staticmethod
    def random_network(arch: Sequence[int], samples: int, use_attention: bool = False):
        """
        Create a random network instance together with training data.

        Returns
        -------
        arch : List[int]
            Architecture list.
        model : GraphQNNGen207
            Randomly initialized network.
        training_data : List[Tuple[Tensor, Tensor]]
            (input, target) pairs generated from the final linear layer.
        target_weight : Tensor
            The weight matrix of the last layer (used for data generation).
        """
        model = GraphQNNGen207(arch, use_attention=use_attention)
        target_weight = model.layers[-1].weight.detach()
        training_data = []
        for _ in range(samples):
            x = torch.randn(arch[0], dtype=torch.float32)
            y = target_weight @ x
            training_data.append((x, y))
        return arch, model, training_data, target_weight

    def feedforward(self, samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
        """
        Run a batch of inputs through the network.

        Parameters
        ----------
        samples : Iterable[Tuple[Tensor, Tensor]]
            Each element is (input, target).  Target is ignored.

        Returns
        -------
        stored : List[List[Tensor]]
            Activation list per sample: [x0, x1,..., xn]
        """
        stored: List[List[Tensor]] = []
        for x, _ in samples:
            activations = [x]
            current = x
            for layer in self.layers:
                current = torch.tanh(layer(current))
                activations.append(current)
                if self.use_attention:
                    # Apply attention on the current activation.
                    current = torch.from_numpy(
                        self.attention.run(
                            rotation_params=current.numpy(),
                            entangle_params=current.numpy(),
                            inputs=current.numpy(),
                        )
                    ).float()
                    activations.append(current)
            stored.append(activations)
        return stored

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        """
        Squared cosine similarity between two classical vectors.
        """
        denom = torch.norm(a) * torch.norm(b) + 1e-12
        return float((a @ b / denom).item() ** 2)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """
        Build a weighted graph from pairwise state fidelities.
        """
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNNGen207.state_fidelity(a, b)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph


__all__ = [
    "GraphQNNGen207",
]
