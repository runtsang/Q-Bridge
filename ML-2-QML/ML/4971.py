"""QuantumKernelMethodGen207 – classical implementation.

This module unifies:
* RBF kernel computation (torch‑based)
* Graph‑based neural network feed‑forward
* Conv and FCL layers with a common API
* Fidelity‑based adjacency graph construction

The class mirrors the quantum interface so that downstream pipelines can swap in the quantum implementation without code changes.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
# 1. Classical kernel utilities
# --------------------------------------------------------------------------- #

class KernalAnsatz(nn.Module):
    """RBF kernel implementation compatible with the quantum interface."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: Tensor, y: Tensor) -> Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """Wrapper that normalises inputs and calls KernalAnsatz."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:  # type: ignore[override]
        return self.ansatz(x.view(1, -1), y.view(1, -1)).squeeze()

def kernel_matrix(a: Sequence[Tensor], b: Sequence[Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute Gram matrix for two datasets using the classical RBF kernel."""
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# --------------------------------------------------------------------------- #
# 2. Graph Neural Network utilities
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

def random_network(qnn_arch: Sequence[int], samples: int) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
    """Return architecture, weights, training data, and target weight."""
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
    """Return activations for each sample across layers."""
    activations: List[List[Tensor]] = []
    for features, _ in samples:
        layerwise = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            layerwise.append(current)
        activations.append(layerwise)
    return activations

def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Squared overlap between two normalized tensors."""
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
    """Build a weighted graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

# --------------------------------------------------------------------------- #
# 3. Convolution and fully‑connected layers
# --------------------------------------------------------------------------- #

def Conv(kernel_size: int = 2, threshold: float = 0.0) -> nn.Module:
    """Return a PyTorch Conv2d filter that mimics a quanvolution layer."""
    class ConvFilter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.kernel_size = kernel_size
            self.threshold = threshold
            self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

        def run(self, data) -> float:
            tensor = torch.as_tensor(data, dtype=torch.float32)
            tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
            logits = self.conv(tensor)
            activations = torch.sigmoid(logits - self.threshold)
            return activations.mean().item()
    return ConvFilter()

def FCL(n_features: int = 1) -> nn.Module:
    """Return a linear layer that mimics a fully‑connected quantum layer."""
    class FullyConnectedLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(n_features, 1)

        def run(self, thetas: Iterable[float]) -> np.ndarray:
            values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
            expectation = torch.tanh(self.linear(values)).mean(dim=0)
            return expectation.detach().numpy()
    return FullyConnectedLayer()

# --------------------------------------------------------------------------- #
# 4. Main hybrid class
# --------------------------------------------------------------------------- #

class QuantumKernelMethodGen207:
    """
    Unified interface that accepts both classical and quantum kernels,
    offers graph‑neural‑network inference, and exposes interchangeable
    convolution and fully‑connected layers.
    """
    def __init__(
        self,
        kernel_type: str = "classical",
        gamma: float = 1.0,
        qnn_arch: Sequence[int] | None = None,
        conv_kwargs: dict | None = None,
        fcl_kwargs: dict | None = None,
    ) -> None:
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.qnn_arch = list(qnn_arch) if qnn_arch else [1, 1]
        self.conv = Conv(**(conv_kwargs or {}))
        self.fcl = FCL(**(fcl_kwargs or {}))

    def compute_kernel(self, a: Sequence[Tensor], b: Sequence[Tensor]) -> np.ndarray:
        if self.kernel_type == "classical":
            return kernel_matrix(a, b, gamma=self.gamma)
        else:
            raise NotImplementedError("Quantum kernel not available in classical build.")

    def run_qnn(self, samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
        _, weights, _, _ = random_network(self.qnn_arch, samples=100)
        return feedforward(self.qnn_arch, weights, samples)

    def build_graph(self, states: Sequence[Tensor], threshold: float) -> nx.Graph:
        return fidelity_adjacency(states, threshold)

    def run_conv(self, data) -> float:
        return self.conv.run(data)

    def run_fcl(self, thetas: Iterable[float]) -> np.ndarray:
        return self.fcl.run(thetas)

__all__ = [
    "KernalAnsatz",
    "Kernel",
    "kernel_matrix",
    "random_network",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
    "Conv",
    "FCL",
    "QuantumKernelMethodGen207",
]
