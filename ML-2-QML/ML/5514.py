"""ConvGen413: Classical convolutional block with optional graph pooling.

This module merges concepts from the original Conv.py, EstimatorQNN.py,
GraphQNN.py, and QCNN.py.  It exposes a single class ConvGen413 that
performs a 2‑D convolution, applies a small fully‑connected network,
and optionally constructs a graph of activations based on cosine
similarity.  The class is fully compatible with PyTorch and can be used
as a drop‑in replacement for the legacy Conv() factory.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
import networkx as nx
from typing import Iterable, Sequence, Tuple

Tensor = torch.Tensor

class ConvGen413(nn.Module):
    """
    A hybrid classical convolutional block that combines:
    * A 2‑D convolution with a learnable bias (like ConvFilter).
    * A small feed‑forward network (like EstimatorQNN).
    * Optional graph‑based pooling of activations (like GraphQNN).
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        hidden_layers: Sequence[int] | None = None,
        use_graph: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        kernel_size : int
            Size of the convolution kernel.
        threshold : float
            Threshold applied to the convolution output before the sigmoid.
        hidden_layers : Sequence[int] | None
            Sizes of the hidden layers in the feed‑forward network.  If
            ``None`` a default architecture [8, 4] is used.
        use_graph : bool
            If ``True`` the forward pass returns a graph of activations
            constructed from pairwise cosine similarity.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.use_graph = use_graph

        # Convolution
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

        # Feed‑forward network
        hidden_layers = hidden_layers or [8, 4]
        layers: list[nn.Module] = []
        in_features = kernel_size * kernel_size
        for h in hidden_layers:
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.Tanh())
            in_features = h
        layers.append(nn.Linear(in_features, 1))
        self.ffn = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor | Tuple[Tensor, nx.Graph]:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor
            2‑D tensor of shape ``(H, W)`` with values in ``[0, 255]``.

        Returns
        -------
        Tensor
            Scalar output of the feed‑forward network.
        nx.Graph
            (Optional) Graph of activations when ``use_graph`` is ``True``.
        """
        # Convolution
        conv_out = self.conv(x.unsqueeze(0).unsqueeze(0))  # (1,1,k,k)
        conv_out = torch.sigmoid(conv_out - self.threshold)
        conv_out = conv_out.view(-1)  # flatten

        out = self.ffn(conv_out)

        if self.use_graph:
            # Build a graph where nodes are hidden units and edges are
            # weighted by cosine similarity of their activations.
            activations = [conv_out]
            current = conv_out
            for layer in self.ffn:
                if isinstance(layer, nn.Linear):
                    current = layer(current)
                    activations.append(current)
                elif isinstance(layer, nn.Tanh):
                    current = torch.tanh(current)
            # Compute pairwise cosine similarity
            G = nx.Graph()
            G.add_nodes_from(range(len(activations)))
            for i, a in enumerate(activations):
                for j, b in enumerate(activations):
                    if i < j:
                        cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
                        G.add_edge(i, j, weight=cos)
            return out.squeeze(), G

        return out.squeeze()

    def run(self, data: np.ndarray) -> float:
        """
        Convenience wrapper that accepts a NumPy array and returns a scalar.

        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape ``(H, W)`` with values in ``[0, 255]``.
        """
        tensor = torch.as_tensor(data, dtype=torch.float32)
        return float(self.forward(tensor))

    @staticmethod
    def random_graph_network(
        qnn_arch: Sequence[int], samples: int
    ) -> Tuple[Sequence[int], list[list[Tensor]], list[Tuple[Tensor, Tensor]], Tensor]:
        """
        Generate a random feed‑forward network and training data.

        Mirrors the utilities in GraphQNN.py.  The returned structure
        contains the architecture, weights, dataset, and target weight.
        """
        weights: list[Tensor] = []
        for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
            weights.append(torch.randn(out_f, in_f, dtype=torch.float32))
        target_weight = weights[-1]
        dataset: list[Tuple[Tensor, Tensor]] = []
        for _ in range(samples):
            features = torch.randn(target_weight.size(1), dtype=torch.float32)
            target = target_weight @ features
            dataset.append((features, target))
        return list(qnn_arch), weights, dataset, target_weight

__all__ = ["ConvGen413"]
