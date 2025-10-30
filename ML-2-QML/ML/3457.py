"""Hybrid classical graph neural network with optional quanvolution.

The class `HybridGraphQNN` encapsulates classical graph neural network
functionality and optionally prepends a quanvolution filter for image
feature extraction.  The implementation extends the original GraphQNN
utilities by adding a quantum‑inspired convolution step and by exposing
the functionality as a single cohesive class.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor

__all__ = [
    "QuanvolutionFilter",
    "QuanvolutionClassifier",
    "HybridGraphQNN",
]

# --------------------------------------------------------------------------- #
#  Classical quanvolution primitives
# --------------------------------------------------------------------------- #

class QuanvolutionFilter(nn.Module):
    """Simple 2‑D convolution that mimics the original `quanvolution` filter.

    The filter operates on a single‑channel image and outputs a flattened
    feature vector.  It is deliberately lightweight so that it can be used
    as a preprocessing step inside a graph neural network.
    """

    def __init__(self) -> None:
        super().__init__()
        # 2×2 kernel, stride 2, producing 4 output channels
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2, bias=False)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        # ``x`` is expected to be (B, 1, H, W)
        features = self.conv(x)               # (B, 4, H/2, W/2)
        return features.view(x.size(0), -1)    # (B, 4 * (H/2) * (W/2))

class QuanvolutionClassifier(nn.Module):
    """A lightweight classifier that uses the quanvolution filter."""

    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        # Assuming 28×28 MNIST style input
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

# --------------------------------------------------------------------------- #
#  Hybrid graph‑neural‑network class
# --------------------------------------------------------------------------- #

class HybridGraphQNN:
    """A classical GNN that can optionally prepend a quanvolution filter.

    Parameters
    ----------
    qnn_arch:
        A sequence of hidden layer widths including input and output.
    use_quanvolution:
        If ``True`` the input image is first transformed by
        :class:`~QuanvolutionFilter` before the linear layers.
    """

    def __init__(
        self,
        qnn_arch: Sequence[int],
        use_quanvolution: bool = False,
    ) -> None:
        self.arch = list(qnn_arch)
        self.use_quanvolution = use_quanvolution
        self.layers: List[nn.Linear] = []

        # Build linear layers
        for in_f, out_f in zip(self.arch[:-1], self.arch[1:]):
            self.layers.append(nn.Linear(in_f, out_f, bias=False))

        # Optional quanvolution filter
        self.qfilter: QuanvolutionFilter | None = (
            QuanvolutionFilter() if use_quanvolution else None
        )

    # --------------------------------------------------------------------- #
    #  Random data / weights generators
    # --------------------------------------------------------------------- #

    @staticmethod
    def _random_weight(in_f: int, out_f: int) -> Tensor:
        """Return a random weight matrix (no bias)."""
        return torch.randn(out_f, in_f, dtype=torch.float32)

    @classmethod
    def random_network(
        cls,
        qnn_arch: Sequence[int],
        samples: int,
    ) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
        """Create a random network and synthetic training data.

        Returns
        -------
        arch, weights, training_data, target_weight
        """
        weights = [cls._random_weight(in_f, out_f)
                   for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:])]
        target_weight = weights[-1]
        training_data = cls.random_training_data(target_weight, samples)
        return list(qnn_arch), weights, training_data, target_weight

    @staticmethod
    def random_training_data(
        weight: Tensor,
        samples: int,
    ) -> List[Tuple[Tensor, Tensor]]:
        """Generate ``samples`` (input, target) pairs using ``weight``."""
        dataset: List[Tuple[Tensor, Tensor]] = []
        for _ in range(samples):
            inp = torch.randn(weight.size(1), dtype=torch.float32)
            tgt = weight @ inp
            dataset.append((inp, tgt))
        return dataset

    # --------------------------------------------------------------------- #
    #  Forward propagation
    # --------------------------------------------------------------------- #

    def feedforward(
        self,
        samples: Iterable[Tuple[Tensor, Tensor]],
    ) -> List[List[Tensor]]:
        """Run the network on a set of samples.

        Each sample is a tuple ``(features, target)``; the target is ignored
        during propagation.  The method returns the activation sequence for
        each sample.
        """
        activations_per_sample: List[List[Tensor]] = []

        for features, _ in samples:
            # Pre‑process with quanvolution if enabled
            if self.use_quanvolution and self.qfilter is not None:
                # ``features`` is expected to be a flat vector; reshape to image
                img = features.view(-1, 1, 28, 28)
                features = self.qfilter(img)

            # Forward through linear layers
            activations: List[Tensor] = [features]
            current = features
            for layer in self.layers:
                current = torch.tanh(layer(current))
                activations.append(current)
            activations_per_sample.append(activations)

        return activations_per_sample

    # --------------------------------------------------------------------- #
    #  Utility functions
    # --------------------------------------------------------------------- #

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        """Squared cosine similarity between two vectors."""
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float(torch.dot(a_norm, b_norm).item() ** 2)

    def fidelity_adjacency(
        self,
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Build a weighted graph from pairwise fidelities of ``states``."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = self.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph
