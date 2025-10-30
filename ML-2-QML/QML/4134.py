"""Quantum implementation of the hybrid classifier using a variational ansatz."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Tuple

import networkx as nx
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf

Tensor = torch.Tensor


class HybridQuantumClassifier(tq.QuantumModule):
    """
    Quantum analogue of the classical HybridQuantumClassifier.
    The graph‑based feature extraction is performed on the classical
    input before encoding it into a quantum state.  A variational circuit
    produces expectation values that replace the classical “quantum”
    linear layer.  The architecture mirrors the PyTorch version so that
    the same class name can be used in both contexts.
    """

    class GraphFeatureEncoder(tq.QuantumModule):
        """Encodes each input feature into a rotation on a qubit."""
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires

        def forward(self, qdev: tq.QuantumDevice, features: Tensor):
            # Encode each of the first n_wires features as an RX rotation.
            for i in range(min(self.n_wires, features.shape[1])):
                tqf.rx(qdev, features[:, i], wires=i)

    class VariationalLayer(tq.QuantumModule):
        """Simple depth‑controlled variational circuit."""
        def __init__(self, n_wires: int, depth: int):
            super().__init__()
            self.n_wires = n_wires
            self.depth = depth
            self.random_layer = tq.RandomLayer(
                n_ops=10 * depth, wires=list(range(n_wires))
            )
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for w in range(self.n_wires):
                self.rx(qdev, wires=w)
                self.ry(qdev, wires=w)
                self.rz(qdev, wires=w)

    def __init__(
        self,
        num_features: int,
        n_wires: int = 4,
        depth: int = 2,
        fidelity_threshold: float = 0.9,
        secondary_threshold: float | None = None,
    ) -> None:
        super().__init__()
        self.fidelity_threshold = fidelity_threshold
        self.secondary_threshold = secondary_threshold
        self.graph_encoder = self.GraphFeatureEncoder(n_wires)
        self.variational = self.VariationalLayer(n_wires, depth)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(n_wires)

    # ------------------------------------------------------------------ #
    #  Fidelity utilities
    # ------------------------------------------------------------------ #
    def _fidelity(self, a: Tensor, b: Tensor) -> float:
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float((a_norm @ b_norm).item() ** 2)

    def _build_fidelity_graph(self, states: Sequence[Tensor]) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
            fid = self._fidelity(a, b)
            if fid >= self.fidelity_threshold:
                graph.add_edge(i, j, weight=1.0)
            elif self.secondary_threshold is not None and fid >= self.secondary_threshold:
                graph.add_edge(i, j, weight=0.5)
        return graph

    def _aggregate_graph_features(self, graph: nx.Graph, states: Sequence[Tensor]) -> Tensor:
        agg = []
        for node in graph.nodes:
            neighbours = list(graph.neighbors(node))
            if neighbours:
                neigh_vec = torch.stack([states[n] for n in neighbours], dim=0).mean(dim=0)
                agg_vec = (states[node] + neigh_vec) / 2
            else:
                agg_vec = states[node]
            agg.append(agg_vec)
        return torch.stack(agg, dim=0)

    # ------------------------------------------------------------------ #
    #  Forward pass
    # ------------------------------------------------------------------ #
    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input batch of shape (batch, num_features).
        """
        graph = self._build_fidelity_graph(x)
        aggregated = self._aggregate_graph_features(graph, x)

        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device)
        self.graph_encoder(qdev, aggregated)
        self.variational(qdev)
        out = self.measure(qdev)
        return self.norm(out)


__all__ = ["HybridQuantumClassifier"]
