from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from typing import Sequence, List, Tuple

# Local imports – adjust the package layout if needed
from.Quanvolution import QuanvolutionFilter
from.GraphQNN import random_network, feedforward, state_fidelity, fidelity_adjacency


class QCNNGen227(nn.Module):
    """
    Classical hybrid model that fuses a quanvolution filter with a graph neural network.
    The filter maps 2×2 image patches to a quantum‑encoded vector; the resulting
    feature vector is propagated through a fully‑connected GNN whose weights are
    sampled from a random network generator.  An adjacency graph can be built from
    the node states using state‑fidelity.
    """

    def __init__(
        self,
        arch: Sequence[int],
        num_samples: int = 100,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.device = torch.device(device)

        # Randomly initialise an architecture‑specific weight set
        self.arch, self.weights, _, self.target_weight = random_network(
            list(arch), num_samples
        )
        self.weights = [w.to(self.device) for w in self.weights]

        self.qfilter = QuanvolutionFilter().to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
        1. Apply quanvolution filter to obtain a flat feature vector.
        2. Feed the features through the GNN defined by ``self.weights``.
        3. Return the final layer output.
        """
        x = x.to(self.device)
        features = self.qfilter(x)          # shape: (batch, 4*14*14)
        return self._gnn_forward(features)

    def _gnn_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Simple feed‑forward using the pre‑generated weights."""
        current = inputs
        for w in self.weights:
            current = torch.tanh(current @ w.t())
        return current

    def fidelity_graph(
        self,
        samples: List[Tuple[torch.Tensor, torch.Tensor]],
        threshold: float = 0.9,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """
        Build a weighted adjacency graph from the node states produced by the
        GNN on the provided samples.  Nodes correspond to individual samples.
        """
        states = [self._gnn_forward(f).detach().cpu() for f, _ in samples]
        return fidelity_adjacency(
            states,
            threshold,
            secondary=secondary,
            secondary_weight=secondary_weight,
        )

    def state_fidelity(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """
        Return the squared cosine similarity between two node states.
        """
        return state_fidelity(a, b)


__all__ = ["QCNNGen227"]
