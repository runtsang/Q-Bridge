"""QuantumClassifierModel: classical hybrid architecture.

This module provides a PyTorch neural network that optionally includes a
convolutional pre‑processor, an RBF kernel for similarity computation,
and a graph‑based adjacency structure based on state fidelities.  The
public interface mirrors the quantum counterpart defined in the QML
module so that the two can be swapped without changing downstream code.
"""

import torch
import torch.nn as nn
import numpy as np
import networkx as nx
from typing import Iterable, Tuple, List

# --------------------------------------------------------------------------- #
# Convolutional filter
# --------------------------------------------------------------------------- #
class ConvFilter(nn.Module):
    """2‑D convolutional filter that emulates a quantum quanvolution layer."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Apply the filter and return a scalar activation."""
        x = data.view(-1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(x)
        return torch.sigmoid(logits - self.threshold).mean(-1)

# --------------------------------------------------------------------------- #
# RBF kernel utilities
# --------------------------------------------------------------------------- #
class RBFKernel(nn.Module):
    """Radial‑basis‑function kernel."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * (diff * diff).sum(dim=-1, keepdim=True))

def kernel_matrix(a: np.ndarray, b: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Return the full Gram matrix for two data arrays."""
    k = RBFKernel(gamma)
    return np.array([[k(torch.tensor(x, dtype=torch.float32),
                       torch.tensor(y, dtype=torch.float32)).item()
                      for y in b] for x in a])

# --------------------------------------------------------------------------- #
# Fidelity‑based graph adjacency
# --------------------------------------------------------------------------- #
def fidelity_adjacency(states: List[torch.Tensor],
                       threshold: float,
                       *,
                       secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    """Construct a weighted graph from pairwise state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for i, state_i in enumerate(states):
        for j in range(i + 1, len(states)):
            state_j = states[j]
            fid = (state_i @ state_j).item() ** 2
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
    return graph

# --------------------------------------------------------------------------- #
# Classical feed‑forward classifier factory
# --------------------------------------------------------------------------- #
def build_classifier_circuit(num_features: int,
                             depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """Build a simple feed‑forward network and return metadata."""
    layers: List[nn.Module] = []
    in_dim = num_features
    weight_sizes: List[int] = []
    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.append(linear)
        layers.append(nn.ReLU())
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features
    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())
    net = nn.Sequential(*layers)
    encoding = list(range(num_features))
    observables = list(range(2))
    return net, encoding, weight_sizes, observables

# --------------------------------------------------------------------------- #
# Composite classifier
# --------------------------------------------------------------------------- #
class QuantumClassifierModel(nn.Module):
    """
    Classical neural network that optionally augments its input with a
    convolutional filter, builds a graph adjacency from training states,
    and exposes an RBF kernel for use in training or inference.
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int],
                 output_dim: int = 2,
                 use_graph: bool = False,
                 kernel_gamma: float = 1.0,
                 conv_kernel_size: int = 2,
                 conv_threshold: float = 0.0,
                 graph_threshold: float = 0.8):
        super().__init__()
        self.use_graph = use_graph
        self.graph_threshold = graph_threshold
        self.kernel_gamma = kernel_gamma

        # Convolution pre‑processor
        self.conv = ConvFilter(conv_kernel_size, conv_threshold)

        # Build feed‑forward layers
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.network = nn.Sequential(*layers)

        # Kernel and optional graph
        self.kernel = RBFKernel(kernel_gamma)
        self.graph: nx.Graph | None = None

    # --------------------------------------------------------------------- #
    def build_graph(self, states: List[torch.Tensor]) -> None:
        """Compute the adjacency graph for a list of state vectors."""
        self.graph = fidelity_adjacency(states, self.graph_threshold)

    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through optional convolution and feed‑forward network."""
        if self.conv is not None:
            conv_out = self.conv(x)
            x = torch.cat([x, conv_out], dim=-1)
        return self.network(x)

    # --------------------------------------------------------------------- #
    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Return the RBF kernel matrix between two batches."""
        return self.kernel(a, b)

    # --------------------------------------------------------------------- #
    def graph_weight(self, i: int, j: int) -> float:
        """Retrieve the graph weight between two indices."""
        if self.graph is None:
            return 1.0
        return self.graph[i][j]["weight"] if self.graph.has_edge(i, j) else 0.0

__all__ = [
    "ConvFilter",
    "RBFKernel",
    "kernel_matrix",
    "fidelity_adjacency",
    "build_classifier_circuit",
    "QuantumClassifierModel",
]
