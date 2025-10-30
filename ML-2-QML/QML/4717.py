"""Hybrid kernel‑regression engine with quantum kernel and graph construction.

This module implements the *quantum* side of the combined architecture.
It provides a variational kernel based on TorchQuantum, a graph builder
using quantum state fidelities, and a regression model that consumes
quantum‑encoded features.
"""

from __future__ import annotations

import itertools
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torchquantum.functional import func_name_dict

__all__ = [
    "QuantumKernelRegressionGraph",
    "VariationalKernel",
    "KernelMatrix",
    "GraphAdjacency",
    "RegressionHead",
]


class VariationalKernel(tq.QuantumModule):
    """Variational quantum kernel using a programmable ansatz.

    The ansatz is a sequence of single‑qubit rotations encoding the
    input vectors.  The kernel value is the absolute overlap of the
    resulting states after encoding ``x`` and ``y``.
    """
    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = tq.QuantumModule._create_from_func_list(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

    @tq.static_support
    def _forward(self, qdev: tq.QuantumDevice,
                 x: torch.Tensor, y: torch.Tensor) -> None:
        # Encode x
        for info in self.ansatz.func_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](qdev, wires=info["wires"], params=params)
        # Encode y with negative parameters
        for info in reversed(self.ansatz.func_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](qdev, wires=info["wires"], params=params)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self._forward(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])


class KernelMatrix:
    """Compute Gram matrix for the variational kernel on the CPU."""

    def __init__(self, kernel: tq.QuantumModule) -> None:
        self.kernel = kernel

    def __call__(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        a_stack = torch.stack(a, dim=0)
        b_stack = torch.stack(b, dim=0)
        return self.kernel(a_stack, b_stack).detach().cpu().numpy()


class GraphAdjacency:
    """Weighted graph from quantum state fidelities.

    Fidelity is the squared magnitude of the overlap between two
    encoded states.  Edges above *threshold* receive weight 1.0.
    """
    def __init__(self, threshold: float = 0.8, secondary: float | None = None,
                 secondary_weight: float = 0.5) -> None:
        self.threshold = threshold
        self.secondary = secondary
        self.secondary_weight = secondary_weight

    def __call__(self, states: Sequence[torch.Tensor]) -> nx.Graph:
        import networkx as nx
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
            fid = self._state_fidelity(s_i, s_j)
            if fid >= self.threshold:
                graph.add_edge(i, j, weight=1.0)
            elif self.secondary is not None and fid >= self.secondary:
                graph.add_edge(i, j, weight=self.secondary_weight)
        return graph

    @staticmethod
    def _state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
        return float((torch.abs(torch.vdot(a, b)) ** 2).item())


class RegressionHead(nn.Module):
    """Regression head for quantum‑encoded features."""
    def __init__(self, input_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class QuantumKernelRegressionGraph(tq.QuantumModule):
    """
    A hybrid model that fuses a variational quantum kernel,
    a fidelity‑based graph, and a regression head.

    Parameters
    ----------
    n_wires : int, optional
        Number of qubits for the variational ansatz.
    """
    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.kernel = VariationalKernel(n_wires)
        self.kernel_matrix = KernelMatrix(self.kernel)
        self.graph_builder = GraphAdjacency(threshold=0.8, secondary=0.5)
        self.head = RegressionHead(input_dim=n_wires)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for quantum kernel regression.

        Parameters
        ----------
        X : torch.Tensor
            Input data of shape (batch_size, n_wires).

        Returns
        -------
        torch.Tensor
            Predicted scalar values of shape (batch_size,).
        """
        # Compute quantum kernel matrix against itself
        kernel_feats = self.kernel(X, X)  # (batch, batch)
        # Build graph from encoded states
        graph = self.graph_builder([kernel_feats[i, i] for i in range(kernel_feats.size(0))])
        # For simplicity, use diagonal kernel values as features
        diag = torch.diagonal(kernel_feats, dim1=0, dim2=1).unsqueeze(-1)
        return self.head(diag)
