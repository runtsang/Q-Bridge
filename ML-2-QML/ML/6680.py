"""GraphQNN: Classical implementation with extended capabilities.

This module extends the original GraphQNN functions by adding:
* Parameter‑efficient weight sharing across layers.
* Batch‑wise forward pass for large datasets.
* Optional noise injection in synthetic data.
* A unified GraphQNN class that exposes all utilities.
"""

import itertools
from typing import Iterable, List, Sequence, Tuple, Dict, Optional

import networkx as nx
import torch
from torch import Tensor

__all__ = ["GraphQNN"]


class GraphQNN:
    """Classical graph neural network with extended utilities."""

    def __init__(self, arch: Sequence[int], weights: Optional[List[Tensor]] = None):
        """
        Parameters
        ----------
        arch : Sequence[int]
            List of layer widths. ``arch[0]`` is the input dimension.
        weights : Optional[List[Tensor]], default=None
            List of weight matrices. If ``None`` random weights are created.
        """
        self.arch = list(arch)
        if weights is None:
            self.weights = [
                self._random_linear(in_f, out_f)
                for in_f, out_f in zip(arch[:-1], arch[1:])
            ]
        else:
            self.weights = weights

    @staticmethod
    def _random_linear(in_features: int, out_features: int) -> Tensor:
        """Generate a random weight matrix with normal distribution."""
        return torch.randn(out_features, in_features, dtype=torch.float32)

    @staticmethod
    def random_training_data(
        weight: Tensor,
        samples: int,
        *,
        noise: float = 0.0,
    ) -> List[Tuple[Tensor, Tensor]]:
        """
        Generate synthetic training pairs ``(x, y)`` where ``y = weight @ x``.
        Optional Gaussian noise can be added to the target.
        """
        dataset: List[Tuple[Tensor, Tensor]] = []
        for _ in range(samples):
            x = torch.randn(weight.size(1), dtype=torch.float32)
            y = weight @ x
            if noise > 0.0:
                y = y + noise * torch.randn_like(y)
            dataset.append((x, y))
        return dataset

    @staticmethod
    def random_network(
        arch: Sequence[int], samples: int
    ) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
        """
        Create a random network and synthetic training data for the last layer.
        Returns the architecture, weight matrices, training dataset, and target weight.
        """
        weights = [
            GraphQNN._random_linear(in_f, out_f)
            for in_f, out_f in zip(arch[:-1], arch[1:])
        ]
        target_weight = weights[-1]
        training_data = GraphQNN.random_training_data(target_weight, samples)
        return list(arch), weights, training_data, target_weight

    def feedforward(
        self, samples: Iterable[Tuple[Tensor, Tensor]]
    ) -> List[List[Tensor]]:
        """
        Perform a forward pass for each sample in ``samples``.
        Returns a list of activation lists for every sample.
        """
        activations: List[List[Tensor]] = []
        for x, _ in samples:
            layer_out = x
            layer_outputs = [x]
            for w in self.weights:
                layer_out = torch.tanh(w @ layer_out)
                layer_outputs.append(layer_out)
            activations.append(layer_outputs)
        return activations

    def batch_feedforward(self, batch: List[Tensor]) -> List[Tensor]:
        """
        Forward pass for a batch of input vectors.
        Returns a list containing the output of the final layer for each input.
        """
        outputs: List[Tensor] = []
        for x in batch:
            out = x
            for w in self.weights:
                out = torch.tanh(w @ out)
            outputs.append(out)
        return outputs

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        """Squared dot product of two normalized vectors."""
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float((a_norm @ b_norm).item() ** 2)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """
        Build a weighted graph where edges represent fidelity greater than
        ``threshold`` (weight 1) or between ``secondary`` and ``threshold``
        (weight ``secondary_weight``).
        """
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNN.state_fidelity(a, b)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    def share_weights(self, layer_indices: List[int]) -> None:
        """
        Share the weight matrix of the first layer across the specified indices.
        Useful for reducing parameter count.
        """
        if not layer_indices:
            return
        base_weight = self.weights[0]
        for idx in layer_indices:
            if 0 <= idx < len(self.weights):
                self.weights[idx] = base_weight
