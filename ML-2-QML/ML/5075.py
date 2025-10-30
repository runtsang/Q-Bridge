import torch
from torch import nn
import numpy as np
import networkx as nx
from typing import Iterable, Sequence, List, Tuple, Optional

Tensor = torch.Tensor

class EstimatorQNN(nn.Module):
    """
    Classical hybrid estimator that unifies regression, classification,
    graph‑based state propagation and a quantum‑style kernel.

    Parameters
    ----------
    architecture : Sequence[int]
        Layer sizes, e.g. [2, 8, 4, 1] for a 2‑input regressor.
    mode : str, optional
        ``"regression"`` or ``"classification"``.  The default is
        ``"regression"``.
    kernel_gamma : float, optional
        RBF kernel bandwidth.  Only relevant if a kernel matrix is
        requested.
    """
    def __init__(self,
                 architecture: Sequence[int],
                 mode: str = "regression",
                 kernel_gamma: float = 1.0):
        super().__init__()
        self.architecture = list(architecture)
        self.mode = mode
        self.kernel_gamma = kernel_gamma

        layers: List[nn.Module] = []
        in_dim = architecture[0]
        activation = nn.Tanh() if mode == "regression" else nn.ReLU()
        for out_dim in architecture[1:]:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(activation)
            in_dim = out_dim

        # Output head
        out_dim = 1 if mode == "regression" else 2
        layers.append(nn.Linear(in_dim, out_dim))

        self.net = nn.Sequential(*layers)

        if mode == "classification":
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        out = self.net(x)
        if self.mode == "classification":
            out = self.softmax(out)
        return out

    # ------------------------------------------------------------------
    # Helper utilities that mirror the reference graph / kernel modules
    # ------------------------------------------------------------------
    @staticmethod
    def _random_linear(in_f: int, out_f: int) -> Tensor:
        return torch.randn(out_f, in_f, dtype=torch.float32)

    @staticmethod
    def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
        data = []
        for _ in range(samples):
            feat = torch.randn(weight.size(1), dtype=torch.float32)
            tgt = weight @ feat
            data.append((feat, tgt))
        return data

    @classmethod
    def random_network(cls, arch: Sequence[int], samples: int):
        weights = []
        for in_f, out_f in zip(arch[:-1], arch[1:]):
            weights.append(cls._random_linear(in_f, out_f))
        target = weights[-1]
        training = cls.random_training_data(target, samples)
        return list(arch), weights, training, target

    def feedforward(self, data: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
        """Return activations of every layer for each sample."""
        activations: List[List[Tensor]] = []
        for feat, _ in data:
            current = feat
            layer_states = [current]
            for layer in self.net:
                current = layer(current)
                layer_states.append(current)
            activations.append(layer_states)
        return activations

    def state_fidelity(self, a: Tensor, b: Tensor) -> float:
        """Squared overlap between two normalized state vectors."""
        a = a / (torch.norm(a) + 1e-12)
        b = b / (torch.norm(b) + 1e-12)
        return float(torch.dot(a, b).item() ** 2)

    def fidelity_adjacency(
        self,
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: Optional[float] = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Build a weighted graph from state fidelities."""
        G = nx.Graph()
        G.add_nodes_from(range(len(states)))
        for i, a in enumerate(states):
            for j, b in enumerate(states[i + 1:], start=i + 1):
                fid = self.state_fidelity(a, b)
                if fid >= threshold:
                    G.add_edge(i, j, weight=1.0)
                elif secondary is not None and fid >= secondary:
                    G.add_edge(i, j, weight=secondary_weight)
        return G

    # ------------------------------------------------------------------
    # Kernel utilities
    # ------------------------------------------------------------------
    class Kernel(nn.Module):
        """RBF kernel that mimics the quantum interface."""
        def __init__(self, gamma: float):
            super().__init__()
            self.gamma = gamma

        def forward(self, x: Tensor, y: Tensor) -> Tensor:
            diff = x - y
            return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

    def kernel_matrix(
        self,
        a: Sequence[Tensor],
        b: Sequence[Tensor],
    ) -> np.ndarray:
        kernel = self.Kernel(self.kernel_gamma)
        return np.array([[kernel(x, y).item() for y in b] for x in a])

__all__ = ["EstimatorQNN"]
