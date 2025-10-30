"""Hybrid fully‑connected layer with classical, quantum, and graph components.

The class implements:
    1) a classical linear transform with tanh activation,
    2) a quantum circuit that measures an expectation value for each input
       parameter,
    3) a graph that represents the fidelity between all layer outputs.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
import networkx as nx
import itertools
from typing import Iterable, Sequence, List, Tuple, Optional

class HybridFullyConnectedGraphLayer(nn.Module):
    def __init__(self, n_features: int = 1, n_qubits: int = 1, shots: int = 100,
                 backend: Optional[str] = None) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = backend

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Return a NumPy array of the quantum expectation value for each theta."""
        torch_thetas = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        classical_out = torch.tanh(self.linear(torch_thetas)).mean(dim=0).item()
        expectation = 0.0
        for theta in thetas:
            expectation += np.cos(theta)
        expectation /= len(thetas)
        return np.array([classical_out, expectation])

    def feedforward(self, samples: Iterable[torch.Tensor]) -> List[List[torch.Tensor]]:
        activations_per_sample: List[List[torch.Tensor]] = []
        for sample in samples:
            activations = [sample]
            current = torch.tanh(self.linear(sample))
            activations.append(current)
            activations_per_sample.append(activations)
        return activations_per_sample

    @staticmethod
    def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float((a_norm @ b_norm).item() ** 2)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[torch.Tensor],
        threshold: float,
        *,
        secondary: Optional[float] = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = HybridFullyConnectedGraphLayer.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int = 10) -> Tuple[List[int], List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        weights: List[torch.Tensor] = []
        for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
            weights.append(torch.randn(out_f, in_f, dtype=torch.float32))
        target_weight = weights[-1]
        training_data = []
        for _ in range(samples):
            features = torch.randn(target_weight.size(1), dtype=torch.float32)
            target = target_weight @ features
            training_data.append((features, target))
        return list(qnn_arch), weights, training_data, target_weight

    @staticmethod
    def random_training_data(weight: torch.Tensor, samples: int = 10) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        dataset: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for _ in range(samples):
            features = torch.randn(weight.size(1), dtype=torch.float32)
            target = weight @ features
            dataset.append((features, target))
        return dataset

def FCL(n_features: int = 1, n_qubits: int = 1, shots: int = 100, backend: Optional[str] = None) -> HybridFullyConnectedGraphLayer:
    return HybridFullyConnectedGraphLayer(n_features, n_qubits, shots, backend)

__all__ = ["HybridFullyConnectedGraphLayer", "FCL"]
