from __future__ import annotations
from typing import Sequence, Iterable, Tuple, List
import numpy as np
import torch
from torch import nn
import networkx as nx
import itertools

# ----------------- classical kernel utilities ----------------- #
class _RBFAnsatz(nn.Module):
    """Radial basis function kernel implemented as a pure PyTorch module."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x.unsqueeze(1) - y.unsqueeze(0)
        return torch.exp(-self.gamma * (diff**2).sum(dim=-1))

class _RBFKernel(nn.Module):
    """Convenience wrapper around :class:`_RBFAnsatz` returning a scalar."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = _RBFAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a: Sequence[torch.Tensor],
                  b: Sequence[torch.Tensor],
                  gamma: float = 1.0) -> np.ndarray:
    """Return the Gram matrix between two collections of samples."""
    kernel = _RBFKernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# ----------------- classical neural‑network utilities ----------------- #
def _random_linear(in_features: int, out_features: int) -> torch.Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: torch.Tensor,
                         samples: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    dataset: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(arch: Sequence[int], samples: int):
    """Generate a toy fully‑connected network and a training set."""
    weights: List[torch.Tensor] = []
    for in_f, out_f in zip(arch[:-1], arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(arch), weights, training_data, target_weight

def feedforward(arch: Sequence[int],
                weights: Sequence[torch.Tensor],
                samples: Iterable[Tuple[torch.Tensor, torch.Tensor]]) -> List[List[torch.Tensor]]:
    """Return the activations of each layer for every sample."""
    activations: List[List[torch.Tensor]] = []
    for features, _ in samples:
        layer_outputs = [features]
        current = features
        for w in weights:
            current = torch.tanh(w @ current)
            layer_outputs.append(current)
        activations.append(layer_outputs)
    return activations

# ----------------- fidelity and graph utilities ----------------- #
def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float((a_norm @ b_norm).item() ** 2)

def fidelity_adjacency(states: Sequence[torch.Tensor],
                       threshold: float,
                       *,
                       secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, ai), (j, aj) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(ai, aj)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

# ----------------- simulated quanvolution ----------------- #
class _SimQuanvolution(nn.Module):
    """Classical 2‑D convolution that mimics the layout used in the
    original quantum quanvolution example."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).view(x.size(0), -1)

# ----------------- hybrid integration ----------------- #
class QuantumKernelMethod(nn.Module):
    """Hybrid kernel‑based module that stitches together classical kernels,
    fully‑connected nets, graph‑based adjacency, and a simulated quanvolution.
    """
    def __init__(self,
                 gamma: float = 1.0,
                 arch: Sequence[int] | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.kernel = _RBFKernel(gamma)
        self.arch = arch or [1, 10, 1]
        self.layers = nn.ModuleList()
        for in_f, out_f in zip(self.arch[:-1], self.arch[1:]):
            self.layers.append(nn.Linear(in_f, out_f))
        self.quanvolution = _SimQuanvolution()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a single sample through the quanvolution filter and the
        fully‑connected backbone."""
        qf = self.quanvolution(x)
        out = qf
        for layer in self.layers:
            out = torch.tanh(layer(out))
        return out

    def kernel_matrix(self,
                      a: Sequence[torch.Tensor],
                      b: Sequence[torch.Tensor]) -> np.ndarray:
        return kernel_matrix(a, b, gamma=self.gamma)

    def random_network(self, samples: int):
        return random_network(self.arch, samples)

    def random_training_data(self, samples: int):
        _, _, data, _ = self.random_network(samples)
        return data

    def feedforward(self,
                    samples: Iterable[Tuple[torch.Tensor, torch.Tensor]]) -> List[List[torch.Tensor]]:
        return feedforward(self.arch, self.layers, samples)

    def fidelity_adjacency(self,
                            states: Sequence[torch.Tensor],
                            threshold: float,
                            *,
                            secondary: float | None = None,
                            secondary_weight: float = 0.5) -> nx.Graph:
        return fidelity_adjacency(states,
                                   threshold,
                                   secondary=secondary,
                                   secondary_weight=secondary_weight)

# Preserve the original public names for backward compatibility.
KernalAnsatz = _RBFAnsatz
Kernel = _RBFKernel
__all__ = [
    "KernalAnsatz",
    "Kernel",
    "kernel_matrix",
    "QuantumKernelMethod",
]
