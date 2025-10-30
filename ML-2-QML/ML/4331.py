import torch
from torch import nn
import networkx as nx
import numpy as np
import itertools
from typing import Sequence, Iterable, Tuple, List

class HybridUnit:
    """Hybrid classical module combining convolution, graph adjacency, estimator and kernel."""
    def __init__(
        self,
        conv_kernel_size: int = 2,
        conv_threshold: float = 0.0,
        graph_threshold: float = 0.9,
        estimator_layers: Sequence[int] = (2, 8, 4, 1),
        kernel_gamma: float = 1.0,
    ) -> None:
        self.conv_kernel_size = conv_kernel_size
        self.conv_threshold = conv_threshold
        self.graph_threshold = graph_threshold
        self.estimator = nn.Sequential(
            nn.Linear(estimator_layers[0], estimator_layers[1]),
            nn.Tanh(),
            nn.Linear(estimator_layers[1], estimator_layers[2]),
            nn.Tanh(),
            nn.Linear(estimator_layers[2], estimator_layers[3]),
        )
        self.kernel_gamma = kernel_gamma

    def conv(self, data: np.ndarray) -> float:
        """Apply a 2D convolution filter and return mean activation."""
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.conv_kernel_size, self.conv_kernel_size)
        conv = nn.Conv2d(1, 1, kernel_size=self.conv_kernel_size, bias=True)
        logits = conv(tensor)
        activations = torch.sigmoid(logits - self.conv_threshold)
        return activations.mean().item()

    def graph_fidelity(self, states: Sequence[torch.Tensor]) -> nx.Graph:
        """Build an adjacency graph based on pairwise state fidelities."""
        def fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
            a_norm = a / (torch.norm(a) + 1e-12)
            b_norm = b / (torch.norm(b) + 1e-12)
            return float(torch.dot(a_norm, b_norm).item() ** 2)

        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
            fid = fidelity(s_i, s_j)
            if fid >= self.graph_threshold:
                graph.add_edge(i, j, weight=1.0)
        return graph

    def estimate(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run the estimator network."""
        return self.estimator(inputs)

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Compute RBF kernel matrix."""
        return np.array(
            [[torch.exp(-self.kernel_gamma * torch.sum((x - y) ** 2)).item() for y in b] for x in a]
        )

__all__ = ["HybridUnit"]
