"""Hybrid graph neural network for classical processing.

This module builds on the original GraphQNN and QuantumKernelMethod
implementations, exposing a single :class:`GraphQNNGen` that can
generate random networks, perform feed‑forward computations and
compute classical RBF kernels.  The API mirrors the seeds while
providing a clean, type‑annotated interface.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple, Optional

import torch
import networkx as nx
import numpy as np

Tensor = torch.Tensor
State = Tensor

class GraphQNNGen:
    """
    Hybrid graph neural network with optional RBF kernel.

    Parameters
    ----------
    arch : Sequence[int]
        Neural‑network widths.
    kernel_type : {"classical", "quantum"}, default "classical"
        Kernel mode; quantum mode raises ``NotImplementedError``.
    gamma : float, default 1.0
        RBF kernel width.
    num_samples : int, default 100
        Number of synthetic training samples.
    device : torch.device | None, default None
        Torch device for computations.
    """

    def __init__(
        self,
        arch: Sequence[int],
        kernel_type: str = "classical",
        gamma: float = 1.0,
        num_samples: int = 100,
        device: Optional[torch.device] = None,
    ) -> None:
        self.arch = list(arch)
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.num_samples = num_samples
        self.device = device or torch.device("cpu")
        self.weights: List[Tensor] = []

    # --------------------------------------------------------------------
    # GraphQNN utilities
    # --------------------------------------------------------------------

    @staticmethod
    def _random_linear(in_features: int, out_features: int) -> Tensor:
        return torch.randn(out_features, in_features, dtype=torch.float32)

    def random_network(self) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
        weights: List[Tensor] = []
        for in_f, out_f in zip(self.arch[:-1], self.arch[1:]):
            weights.append(self._random_linear(in_f, out_f))
        self.weights = weights
        target_weight = weights[-1]
        training_data = self.random_training_data(target_weight, self.num_samples)
        return self.arch, weights, training_data, target_weight

    def random_training_data(self, weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
        dataset: List[Tuple[Tensor, Tensor]] = []
        for _ in range(samples):
            features = torch.randn(weight.size(1), dtype=torch.float32, device=self.device)
            target = weight @ features
            dataset.append((features, target))
        return dataset

    def feedforward(
        self,
        samples: Iterable[Tuple[Tensor, Tensor]],
    ) -> List[List[Tensor]]:
        stored: List[List[Tensor]] = []
        for features, _ in samples:
            activations = [features]
            current = features
            for weight in self.weights:
                current = torch.tanh(weight @ current)
                activations.append(current)
            stored.append(activations)
        return stored

    @staticmethod
    def state_fidelity(a: State, b: State) -> float:
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float(torch.dot(a_norm, b_norm).item() ** 2)

    def fidelity_adjacency(
        self,
        states: Sequence[State],
        threshold: float,
        *,
        secondary: Optional[float] = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = self.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    # --------------------------------------------------------------------
    # Kernel utilities
    # --------------------------------------------------------------------

    def kernel_matrix(
        self,
        a: Sequence[Tensor],
        b: Sequence[Tensor],
        gamma: Optional[float] = None,
    ) -> np.ndarray:
        """Return Gram matrix between two collections of samples.

        For ``kernel_type == "classical"`` a radial‑basis‑function kernel
        is used.  The quantum variant is a stub that raises
        ``NotImplementedError``.
        """
        if self.kernel_type == "classical":
            g = gamma if gamma is not None else self.gamma
            a_tensor = torch.stack(list(a))
            b_tensor = torch.stack(list(b))
            diff = a_tensor[:, None] - b_tensor[None, :]
            kernel = torch.exp(-g * torch.sum(diff * diff, dim=-1, keepdim=True))
            return kernel.cpu().numpy()
        raise NotImplementedError("Quantum kernel not implemented in classical module")

__all__ = [
    "GraphQNNGen",
]
