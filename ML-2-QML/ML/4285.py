"""Hybrid classical graph neural network inspired by Quantum‑NAT.

The model encodes 2‑D inputs with a lightweight CNN, then propagates
through a fully‑connected graph of hidden layers.  The graph
architecture is instantiated from a list of layer sizes.  A
fidelity‑based adjacency graph can be constructed from the
layer activations, enabling graph‑based analysis of the
representation flow.  The class is compatible with the
FastBaseEstimator interface, providing a batched evaluation
method.

The implementation deliberately extends the original QFCModel
by adding a graph propagation stage, a fidelity graph helper
and a convenient evaluate routine that mirrors the
FastBaseEstimator logic.
"""

from __future__ import annotations

import itertools
from typing import Callable, Iterable, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np


ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


class QuantumNATHybrid(nn.Module):
    """Classical CNN + graph neural network hybrid."""

    def __init__(
        self,
        cnn_channels: Sequence[int] = (8, 16),
        graph_arch: Sequence[int] = (64, 32, 4),
    ) -> None:
        super().__init__()
        # --- CNN encoder ----------------------------------------------------
        layers: List[nn.Module] = []
        in_ch = 1
        for out_ch in cnn_channels:
            layers.extend(
                [
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                ]
            )
            in_ch = out_ch
        self.encoder = nn.Sequential(*layers)

        # --- Graph network --------------------------------------------------
        self.weights: nn.ModuleList = nn.ModuleList()
        for in_f, out_f in zip(graph_arch[:-1], graph_arch[1:]):
            self.weights.append(nn.Linear(in_f, out_f))
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the final graph layer output."""
        bsz = x.shape[0]
        feats = self.encoder(x)
        feats = feats.view(bsz, -1)
        h = feats
        for w in self.weights:
            h = self.activation(w(h))
        return h

    # -----------------------------------------------------------------------
    # Graph utilities
    # -----------------------------------------------------------------------
    def fidelity_graph(
        self,
        activations: List[torch.Tensor],
        threshold: float = 0.9,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """
        Build a weighted adjacency graph from layer activations.

        Nodes correspond to each layer activation.  Edges are weighted
        by the squared cosine similarity (fidelity) between the
        flattened activations.  Edges above ``threshold`` receive
        weight 1.0 and, if ``secondary`` is provided, edges
        between ``secondary`` and ``threshold`` receive
        ``secondary_weight``.
        """
        graph = nx.Graph()
        graph.add_nodes_from(range(len(activations)))
        for (i, a), (j, b) in itertools.combinations(enumerate(activations), 2):
            a_flat = a.view(-1)
            b_flat = b.view(-1)
            fid = torch.dot(a_flat, b_flat) / (
                torch.norm(a_flat) * torch.norm(b_flat) + 1e-12
            )
            fid = fid.item() ** 2
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    # -----------------------------------------------------------------------
    # Estimator utilities
    # -----------------------------------------------------------------------
    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """
        Evaluate a collection of scalar observables on batches of
        input parameters, mirroring the FastBaseEstimator interface.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

        self.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = torch.as_tensor(params, dtype=torch.float32)
                if inputs.ndim == 1:
                    inputs = inputs.unsqueeze(0)
                outputs = self(inputs)
                row: List[float] = []
                for obs in observables:
                    value = obs(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                results.append(row)
        return results


__all__ = ["QuantumNATHybrid"]
