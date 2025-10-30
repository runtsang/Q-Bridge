"""Hybrid kernel network with classical implementations."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
import networkx as nx
import itertools
from typing import Sequence, Iterable, List, Tuple

# Classical radial‑basis function kernel
class RBFKernel(nn.Module):
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

def kernel_matrix_classical(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    kernel = RBFKernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# Classical fully‑connected layer placeholder
class FCL(nn.Module):
    def __init__(self, n_features: int = 1):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.linear(values)).mean(dim=0)
        return expectation.detach().numpy()

# Graph‑based neural network utilities
def _random_linear(in_features: int, out_features: int) -> torch.Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: torch.Tensor, samples: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    dataset = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    weights = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight

def feedforward_graph(qnn_arch: Sequence[int], weights: Sequence[torch.Tensor],
                      samples: Iterable[Tuple[torch.Tensor, torch.Tensor]]) -> List[List[torch.Tensor]]:
    stored = []
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

def fidelity_adjacency(states: Sequence[torch.Tensor], threshold: float,
                       *, secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

# Sampler neural network
class SamplerQNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(inputs), dim=-1)

# Unified hybrid class
class HybridKernelNetwork:
    """
    A unified interface that can operate in either classical or quantum mode.
    Classical mode uses NumPy/Torch implementations; quantum mode delegates
    to the corresponding qml implementations.
    """

    def __init__(self, mode: str = "classical", **kwargs):
        if mode not in {"classical", "quantum"}:
            raise ValueError("mode must be 'classical' or 'quantum'")
        self.mode = mode
        if mode == "classical":
            self.kernel = lambda a, b, gamma=1.0: kernel_matrix_classical(a, b, gamma)
            self.fcl = FCL()
            self.graph = {
                "random_network": random_network,
                "feedforward": feedforward_graph,
                "fidelity_adjacency": fidelity_adjacency,
            }
            self.sampler = SamplerQNN()
        else:
            # placeholders; quantum implementations are defined in the qml module
            self.kernel = None
            self.fcl = None
            self.graph = None
            self.sampler = None

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
        if self.mode!= "classical":
            raise RuntimeError("Quantum kernel must be used from the qml module.")
        return self.kernel(a, b, gamma)

    def run_fcl(self, thetas: Iterable[float]) -> np.ndarray:
        if self.mode!= "classical":
            raise RuntimeError("Quantum FCL must be used from the qml module.")
        return self.fcl.run(thetas)

    def graph_feedforward(self, samples: Iterable[Tuple[torch.Tensor, torch.Tensor]]) -> List[List[torch.Tensor]]:
        if self.mode!= "classical":
            raise RuntimeError("Quantum graph feedforward must be used from the qml module.")
        arch, weights, _, _ = random_network([1, 2, 1], len(samples))
        return self.graph["feedforward"](arch, weights, samples)

    def sample(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.mode!= "classical":
            raise RuntimeError("Quantum sampler must be used from the qml module.")
        return self.sampler(inputs)
