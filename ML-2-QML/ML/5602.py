"""Hybrid classical/quantum utilities for kernel methods and neural networks.

This module provides a classical implementation of the original
``QuantumKernelMethod`` interface and extends it with hybrid
capabilities.  It can optionally incorporate quantum kernels,
classifiers, estimators, and graph‑based QNNs supplied by the
corresponding QML module.  All classes are fully importable and
expose a ``__call__`` or ``forward`` method for use in training
pipelines.
"""

from __future__ import annotations

import itertools
from typing import Iterable, Sequence, Tuple, List, Any

import numpy as np
import torch
import torch.nn as nn
import networkx as nx

# --------------------------------------------------------------------------- #
# Classical kernel
# --------------------------------------------------------------------------- #
class KernalAnsatz(nn.Module):
    """Classical radial basis function kernel."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """Wrapper around :class:`KernalAnsatz` that accepts batched inputs."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# --------------------------------------------------------------------------- #
# Hybrid kernel
# --------------------------------------------------------------------------- #
class HybridKernel(nn.Module):
    """Hybrid RBF + quantum kernel.

    Parameters
    ----------
    gamma : float
        Width of the classical RBF kernel.
    quantum_weight : float
        Weight of the quantum kernel in [0, 1].
    quantum_kernel_callable : Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None
        Optional callable that evaluates the quantum kernel.  If not
        provided, a zero kernel is used.
    """
    def __init__(
        self,
        gamma: float = 1.0,
        quantum_weight: float = 0.5,
        quantum_kernel_callable: Any = None,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.quantum_weight = quantum_weight
        self.classical = Kernel(gamma)
        self.quantum_kernel = quantum_kernel_callable

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        classical = self.classical(x, y)
        quantum = torch.zeros_like(classical)
        if self.quantum_kernel is not None:
            quantum = self.quantum_kernel(x, y)
        return (1.0 - self.quantum_weight) * classical + self.quantum_weight * quantum

# --------------------------------------------------------------------------- #
# Classical classifier factory
# --------------------------------------------------------------------------- #
def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """Return a classical feed‑forward network and ancillary metadata."""
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
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
    network = nn.Sequential(*layers)
    observables = list(range(2))
    return network, encoding, weight_sizes, observables

# --------------------------------------------------------------------------- #
# Classical estimator
# --------------------------------------------------------------------------- #
def EstimatorQNN() -> nn.Module:
    """Return a small fully‑connected regression network."""
    class EstimatorNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 8),
                nn.Tanh(),
                nn.Linear(8, 4),
                nn.Tanh(),
                nn.Linear(4, 1),
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return self.net(inputs)

    return EstimatorNet()

# --------------------------------------------------------------------------- #
# Graph‑based utilities (classical)
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> torch.Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: torch.Tensor, samples: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    dataset: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    weights: List[torch.Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight

def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[torch.Tensor],
    samples: Iterable[Tuple[torch.Tensor, torch.Tensor]],
) -> List[List[torch.Tensor]]:
    stored: List[List[torch.Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        stored.append(activations)
    return stored

def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)

def fidelity_adjacency(
    states: Sequence[torch.Tensor],
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

# --------------------------------------------------------------------------- #
# Hybrid wrappers
# --------------------------------------------------------------------------- #
class HybridClassifier(nn.Module):
    """Combines the classical classifier backbone with a quantum layer."""
    def __init__(
        self,
        num_features: int,
        depth: int,
        quantum_classifier_callable: Any = None,
    ) -> None:
        super().__init__()
        self.classical, self.enc, self.w_sizes, _ = build_classifier_circuit(num_features, depth)
        self.quantum_classifier = quantum_classifier_callable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        classical_logits = self.classical(x)
        if self.quantum_classifier is None:
            return classical_logits
        quantum_out = self.quantum_classifier(x)
        return torch.cat([classical_logits, quantum_out], dim=-1)

class HybridEstimatorQNN(nn.Module):
    """Wraps the classical estimator with an optional quantum estimator."""
    def __init__(self, quantum_estimator_callable: Any = None) -> None:
        super().__init__()
        self.classical = EstimatorQNN()
        self.quantum_estimator = quantum_estimator_callable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.classical(x)
        if self.quantum_estimator is not None:
            out = out + self.quantum_estimator(x)
        return out

class HybridGraphQNN:
    """Utility class that can build either a classical or quantum graph‑based QNN."""
    def __init__(self, use_quantum: bool = False) -> None:
        self.use_quantum = use_quantum

    def build(self, qnn_arch: Sequence[int], samples: int):
        if self.use_quantum:
            raise NotImplementedError("Quantum graph utilities require the QML module.")
        return random_network(qnn_arch, samples)

    def feedforward(
        self,
        qnn_arch: Sequence[int],
        weights: Sequence[torch.Tensor],
        samples: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    ) -> List[List[torch.Tensor]]:
        if self.use_quantum:
            raise NotImplementedError("Quantum graph utilities require the QML module.")
        return feedforward(qnn_arch, weights, samples)

    def adjacency(
        self,
        states: Sequence[torch.Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        if self.use_quantum:
            raise NotImplementedError("Quantum graph utilities require the QML module.")
        return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)

# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
class QuantumKernelHybrid:
    """Unified class that bundles classical and quantum kernels, classifiers,
    estimators, and graph utilities."""
    def __init__(self, gamma: float = 1.0, quantum_weight: float = 0.5, n_wires: int = 4):
        self.kernel = HybridKernel(gamma, quantum_weight)
        self.classifier = None
        self.estimator = None
        self.graph_util = None

    def set_classifier(self, num_features: int, depth: int, quantum_classifier_callable=None):
        self.classifier = HybridClassifier(num_features, depth, quantum_classifier_callable)

    def set_estimator(self, quantum_estimator_callable=None):
        self.estimator = HybridEstimatorQNN(quantum_estimator_callable)

    def set_graph_util(self, use_quantum=False):
        self.graph_util = HybridGraphQNN(use_quantum)
