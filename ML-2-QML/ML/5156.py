"""GraphQNNHybrid – classical implementation with graph and convolutional layers.

The class mirrors the structure of the original GraphQNN but adds
optional 2×2 convolutional feature extraction (either a simple
torch.Conv2d or a custom ConvFilter) and a very lightweight graph
neural network that aggregates neighbor activations using the
fidelity‑based adjacency graph.

The public API is intentionally identical to the QML counterpart:
```
model = GraphQNNHybrid(arch=[4,8,4])
train_data = random_network(arch, samples=100)
activations = feedforward(arch, weights, train_data)
```
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
#  Convolutional filter – classical 2×2 kernel
# --------------------------------------------------------------------------- #
class ConvFilter(nn.Module):
    """Simple 2×2 convolution followed by a sigmoid and a mean.

    The filter is equivalent to the “Conv” helper in the original
    project but implemented purely with torch ops so it can be used
    inside a larger network.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        self.threshold = threshold

    def forward(self, x: Tensor) -> Tensor:
        # x is (batch, 1, H, W)
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=[2, 3])  # collapse spatial dims

# --------------------------------------------------------------------------- #
#  GraphQNNHybrid – main module
# --------------------------------------------------------------------------- #
class GraphQNNHybrid(nn.Module):
    """Hybrid graph neural network with optional convolutional feature extractor.

    Parameters
    ----------
    arch : Sequence[int]
        Layer sizes for the fully connected graph network.
    use_conv : bool, default=True
        Whether to prepend a 2×2 convolutional filter.
    conv_kernel : int, default=2
        Size of the convolutional kernel.
    conv_threshold : float, default=0.0
        Threshold for the sigmoid in the conv layer.
    """
    def __init__(
        self,
        arch: Sequence[int],
        use_conv: bool = True,
        conv_kernel: int = 2,
        conv_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.arch = list(arch)
        self.use_conv = use_conv

        # Convolutional front‑end
        if self.use_conv:
            self.conv = ConvFilter(conv_kernel, conv_threshold)
        else:
            self.conv = nn.Identity()

        # Graph neural network core
        layers: List[nn.Module] = []
        for in_f, out_f in zip(self.arch[:-1], self.arch[1:]):
            layers.append(nn.Linear(in_f, out_f))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(self.arch[-1], 1))
        self.gnn = nn.Sequential(*layers)

    # --------------------------------------------------------------------- #
    #  Forward pass – returns node‑wise activations after each layer
    # --------------------------------------------------------------------- #
    def forward(self, data: Tensor, graph: nx.Graph) -> List[Tensor]:
        """
        Parameters
        ----------
        data : Tensor
            Input image tensor of shape (B, 1, H, W).
        graph : nx.Graph
            Adjacency graph over the batch dimension.  Each node
            corresponds to a sample in the batch.
        Returns
        -------
        activations : List[Tensor]
            List of node feature tensors after each GNN layer.
        """
        # Step 1 – convolution
        features = self.conv(data)  # (B, 1)
        features = features.squeeze(1)  # (B,)

        # Step 2 – graph propagation
        activations: List[Tensor] = [features]
        current = features
        for layer in self.gnn:
            if isinstance(layer, nn.Linear):
                # Linear transformation on node features
                current = layer(current)
            elif isinstance(layer, nn.Tanh):
                # Message passing: weighted sum of neighbor activations
                # Simple implementation: mean of neighbors
                neighbor_sums = torch.zeros_like(current)
                for i in graph.nodes:
                    neighbors = list(graph.neighbors(i))
                    if neighbors:
                        neighbor_sums[i] = current[neighbors].mean()
                current = layer(current + neighbor_sums)
            else:
                # Skip other modules (e.g., final Linear)
                current = layer(current)
            activations.append(current)
        return activations

# --------------------------------------------------------------------------- #
#  Utility functions – mirror the original GraphQNN helpers
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    weights: List[Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight

def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    stored: List[List[Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        stored.append(activations)
    return stored

def state_fidelity(a: Tensor, b: Tensor) -> float:
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)

def fidelity_adjacency(
    states: Sequence[Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

__all__ = [
    "GraphQNNHybrid",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
