import numpy as np
import torch
import networkx as nx
import itertools
from typing import List, Tuple, Sequence, Iterable, Optional

class QuantumKernelMethod:
    """Classical kernel interface supporting RBF and graph kernels."""
    def __init__(self, kernel_type: str = "rbf", gamma: float = 1.0, graph_threshold: float = 0.8):
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.graph_threshold = graph_threshold

    def _rbf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

    def _graph_kernel(self, idx_x: int, idx_y: int, adjacency: nx.Graph) -> torch.Tensor:
        weight = adjacency.get_edge_data(idx_x, idx_y, default={"weight": 0.0})["weight"]
        return torch.tensor(weight, dtype=torch.float32)

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], adjacency: Optional[nx.Graph] = None) -> np.ndarray:
        if self.kernel_type == "rbf":
            return np.array([[self._rbf(x, y).item() for y in b] for x in a])
        elif self.kernel_type == "graph":
            if adjacency is None:
                raise ValueError("Adjacency graph required for graph kernel")
            return np.array([[self._graph_kernel(i, j, adjacency).item() for j in range(len(b))] for i in range(len(a))])
        else:
            raise ValueError(f"Unsupported kernel type {self.kernel_type}")

    # ------------------------------------------------------------------
    # Graph QNN utilities (classical analogues of the QML version)
    # ------------------------------------------------------------------
    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int) -> Tuple[List[int], List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        weights = []
        for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
            weights.append(torch.randn(out_f, in_f))
        target_weight = weights[-1]
        training_data = []
        for _ in range(samples):
            features = torch.randn(target_weight.size(1))
            target = target_weight @ features
            training_data.append((features, target))
        return list(qnn_arch), weights, training_data, target_weight

    @staticmethod
    def feedforward(qnn_arch: Sequence[int], weights: Sequence[torch.Tensor], samples: List[Tuple[torch.Tensor, torch.Tensor]]) -> List[List[torch.Tensor]]:
        stored = []
        for features, _ in samples:
            activations = [features]
            current = features
            for weight in weights:
                current = torch.tanh(weight @ current)
                activations.append(current)
            stored.append(activations)
        return stored

    @staticmethod
    def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float(torch.dot(a_norm, b_norm).item() ** 2)

    @staticmethod
    def fidelity_adjacency(states: Sequence[torch.Tensor], threshold: float, *, secondary: Optional[float] = None, secondary_weight: float = 0.5) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = QuantumKernelMethod.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

__all__ = ["QuantumKernelMethod"]
