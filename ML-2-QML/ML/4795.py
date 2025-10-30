"""Hybrid classical kernel + graph neural network utilities.

This module retains the legacy names `KernalAnsatz`, `Kernel` and `kernel_matrix`
for backward compatibility, while adding a fully‑connected layer surrogate and
graph‑based neural utilities.  The composite class `QuantumKernelGraphNet`
combines these components and optionally augments the classical kernel with
a simple TorchQuantum ansatz when requested.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import networkx as nx

# --------------------------------------------------------------------------- #
# 1. Classical kernel utilities
# --------------------------------------------------------------------------- #
class KernalAnsatz(nn.Module):
    """Classical RBF kernel with optional scaling.

    The `scale` parameter allows the kernel value to be modulated when a
    quantum component already contributes a similarity score.
    """
    def __init__(self, gamma: float = 1.0, scale: float | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.scale = scale

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        kernel_val = torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))
        if self.scale is not None:
            kernel_val *= self.scale
        return kernel_val


class Kernel(nn.Module):
    """Wraps a :class:`KernalAnsatz` and exposes a legacy API."""
    def __init__(self, gamma: float = 1.0, scale: float | None = None) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma, scale)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Legacy code expects 1‑D vectors; normalise to 2‑D
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor],
                  gamma: float = 1.0, scale: float | None = None) -> np.ndarray:
    """Build a Gram matrix from two lists of 1‑D tensors."""
    kernel = Kernel(gamma, scale)
    return np.array([[kernel(x, y).item() for y in b] for x in a])


# --------------------------------------------------------------------------- #
# 2. Fully connected layer stand‑in
# --------------------------------------------------------------------------- #
class FCL(nn.Module):
    """Classical surrogate for a parameterised quantum fully‑connected layer."""
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        theta_tensor = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        return torch.tanh(self.linear(theta_tensor)).mean(dim=0).detach().numpy()


# --------------------------------------------------------------------------- #
# 3. Graph‑based neural utilities
# --------------------------------------------------------------------------- #
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
    weights = [_random_linear(in_f, out_f) for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:])]
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight


def feedforward(qnn_arch: Sequence[int], weights: Sequence[torch.Tensor],
                samples: Iterable[Tuple[torch.Tensor, torch.Tensor]]) -> List[List[torch.Tensor]]:
    stored = []
    for features, _ in samples:
        activations = [features]
        current = features
        for w in weights:
            current = torch.tanh(w @ current)
            activations.append(current)
        stored.append(activations)
    return stored


def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float((a_norm @ b_norm).item() ** 2)


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


# --------------------------------------------------------------------------- #
# 4. Hybrid composite class
# --------------------------------------------------------------------------- #
# Try to import TorchQuantum; fall back if unavailable.
try:
    import torchquantum as tq
    from torchquantum.functional import func_name_dict
except Exception:
    tq = None
    func_name_dict = {}


class QuantumKernelGraphNet(nn.Module):
    """
    Composite module that provides:
      * a classical or hybrid quantum kernel
      * a fully‑connected layer head
      * graph‑based message passing over state fidelities
    The API mirrors the classical version for ease of comparison.
    """
    def __init__(self,
                 gamma: float = 1.0,
                 kernel_scale: float | None = None,
                 n_features: int = 1,
                 n_wires: int = 4,
                 use_quantum: bool = False):
        super().__init__()
        self.gamma = gamma
        self.kernel_scale = kernel_scale
        self.n_features = n_features
        self.n_wires = n_wires
        self.use_quantum = use_quantum

        # Kernel
        self.kernel = Kernel(gamma, kernel_scale)

        # Fully‑connected layer
        self.fcl = FCL(n_features)

        # Optional quantum kernel fallback
        if use_quantum and tq is not None:
            self.q_device = tq.QuantumDevice(n_wires=n_wires)
            self.q_ansatz = self._build_quantum_ansatz()
        else:
            self.q_device = None
            self.q_ansatz = None

    def _build_quantum_ansatz(self):
        """Return a simple Ry ansatz – one gate per wire."""
        return [
            {"input_idx": [i], "func": "ry", "wires": [i]} for i in range(self.n_wires)
        ]

    def kernel_value(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the (possibly hybrid) kernel value."""
        base = self.kernel(x, y)
        if self.q_ansatz and self.q_device and tq is not None:
            # Encode x and y into quantum device and evaluate overlap
            self.q_device.reset_states(1)
            for info in self.q_ansatz:
                params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
                func_name_dict[info["func"]](self.q_device, wires=info["wires"], params=params)
            for info in reversed(self.q_ansatz):
                params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
                func_name_dict[info["func"]](self.q_device, wires=info["wires"], params=params)
            quantum_overlap = torch.abs(self.q_device.states.view(-1)[0])
            return base * quantum_overlap
        return base

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return np.array([[self.kernel_value(x, y).item() for y in b] for x in a])

    def run_fcl(self, thetas: Iterable[float]) -> np.ndarray:
        return self.fcl.run(thetas)

    def random_graph_network(self, qnn_arch: Sequence[int], samples: int):
        return random_network(qnn_arch, samples)

    def feedforward(self, qnn_arch: Sequence[int], weights: Sequence[torch.Tensor],
                    samples: Iterable[Tuple[torch.Tensor, torch.Tensor]]):
        return feedforward(qnn_arch, weights, samples)

    def fidelity_graph(self, states: Sequence[torch.Tensor], threshold: float,
                       *, secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
        return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)


__all__ = [
    "KernalAnsatz",
    "Kernel",
    "kernel_matrix",
    "FCL",
    "random_network",
    "random_training_data",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
    "QuantumKernelGraphNet",
]
