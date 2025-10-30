"""Unified classical estimator module.

Provides a simple feed‑forward neural network that can be wrapped by a quantum estimator or used directly for regression.  The network supports weight initialization from a graph adjacency matrix, enabling a seamless transition from classical graph to quantum circuits.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import networkx as nx
from typing import Iterable, Sequence, List, Optional, Tuple

Tensor = torch.Tensor
WeightMatrix = List[Tensor]

class UnifiedEstimatorQNN(nn.Module):
    """A lightweight feed‑forward regressor with optional graph‑derived weight initialization."""

    def __init__(
        self,
        architecture: Sequence[int],
        init_weights: Optional[WeightMatrix] = None,
        graph: Optional[nx.Graph] = None,
    ) -> None:
        """
        Parameters
        ----------
        architecture
            1‑D sequence of layer sizes.  The first element is the input
            dimensionality, the last element is the output size.
        init_weights
            Optional list of matrices to initialize the linear layers.
            If ``None`` a random normal initialization is used.
        graph
            Optional graph whose adjacency matrix is converted to a
            weight matrix that matches ``architecture``.  The conversion
            uses a simple linear scaling; it is primarily meant to
            illustrate the hybrid workflow.
        """
        super().__init__()
        self.arch = list(architecture)

        # Build layers
        layers: List[nn.Module] = []
        for in_dim, out_dim in zip(self.arch[:-1], self.arch[1:]):
            layers.append(nn.Linear(in_dim, out_dim, bias=True))
            layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)

        if init_weights is not None:
            self._load_weights(init_weights)
        elif graph is not None:
            gm = self._graph_to_weight_matrix(graph, self.arch[1:])
            self._load_weights(gm)

    def _load_weights(self, weights: WeightMatrix) -> None:
        """Load weight matrices into the network; biases are zeroed."""
        for layer, w in zip(self.net, self.net[1::2]):
            if isinstance(layer, nn.Linear):
                layer.weight.data.copy_(torch.tensor(w, dtype=torch.float32))
                if layer.bias is not None:
                    layer.bias.data.zero_()

    def _graph_to_weight_matrix(
        self,
        graph: nx.Graph,
        output_dims: Sequence[int],
    ) -> WeightMatrix:
        """Return a list of weight matrices derived from ``graph``.
        The adjacency matrix is first normalised and then split into blocks
        that match the requested layer widths.
        """
        adj = nx.to_numpy_array(graph)
        # Normalise
        adj = adj / (adj.max() + 1e-12)
        # Flatten and truncate to match total weight count
        total_weights = sum(in_dim * out_dim for in_dim, out_dim in zip(self.arch[:-1], self.arch[1:]))
        flat = adj.flatten()[:total_weights]
        matrices: WeightMatrix = []
        idx = 0
        for in_dim, out_dim in zip(self.arch[:-1], self.arch[1:]):
            size = in_dim * out_dim
            block = flat[idx : idx + size]
            matrices.append(block.reshape(out_dim, in_dim))
            idx += size
        return matrices

    def forward(self, x: Tensor) -> Tensor:
        """Standard forward pass."""
        return self.net(x)

    # --------------------------------------------------------------------- #
    #  Utility functions that mirror the GraphQNN helpers
    # --------------------------------------------------------------------- #

    @staticmethod
    def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
        """Generate synthetic data for a linear mapping."""
        dataset: List[Tuple[Tensor, Tensor]] = []
        for _ in range(samples):
            features = torch.randn(weight.size(1))
            target = weight @ features
            dataset.append((features, target))
        return dataset

    @staticmethod
    def feedforward(
        architecture: Sequence[int],
        weights: Sequence[Tensor],
        samples: Iterable[Tuple[Tensor, Tensor]],
    ) -> List[List[Tensor]]:
        """Forward pass that records activations at each layer."""
        activations: List[List[Tensor]] = []
        for features, _ in samples:
            layer_outputs = [features]
            current = features
            for w in weights:
                current = torch.tanh(w @ current)
                layer_outputs.append(current)
            activations.append(layer_outputs)
        return activations

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        """Squared overlap between two normalized vectors."""
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float((a_norm @ b_norm).item() ** 2)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: Optional[float] = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Create a graph where edges represent state fidelities."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for i, a in enumerate(states):
            for j, b in enumerate(states[i + 1 :], start=i + 1):
                fid = UnifiedEstimatorQNN.state_fidelity(a, b)
                if fid >= threshold:
                    graph.add_edge(i, j, weight=1.0)
                elif secondary is not None and fid >= secondary:
                    graph.add_edge(i, j, weight=secondary_weight)
        return graph

__all__ = ["UnifiedEstimatorQNN"]
