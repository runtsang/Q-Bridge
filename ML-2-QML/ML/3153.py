"""Hybrid classical‑quantum graph layer (classical implementation).

This module defines a single class, `HybridGraphQLayer`, that:
* implements a classical fully‑connected layer using PyTorch,
* simulates a simple quantum expectation value (cosine of a rotation angle) for each node,
* builds a weighted graph from the fidelities of the simulated quantum states,
* exposes a `run` method returning the combined classical‑quantum output and the graph.
The design keeps the interface identical to the original `FCL.py` so that downstream code can import it unchanged.

The quantum part is simulated classically to keep the module entirely classical.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
import torch
from torch import nn
import networkx as nx

# --------------------------------------------------------------------------- #
# Classical component – fully‑connected layer
# --------------------------------------------------------------------------- #
class _ClassicFCL(nn.Module):
    """Simple linear layer with a tanh activation."""
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the tanh of the linear output."""
        return torch.tanh(self.linear(x))

# --------------------------------------------------------------------------- #
# Quantum component – classical simulation
# --------------------------------------------------------------------------- #
def _quantum_expectation(thetas: Iterable[float]) -> np.ndarray:
    """Simulate expectation of Z after Ry(theta) on |0> (classically)."""
    return np.array([np.cos(theta) for theta in thetas])

# --------------------------------------------------------------------------- #
# Fidelity and graph utilities
# --------------------------------------------------------------------------- #
def _state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
    """Return the squared overlap between two real vectors."""
    a_norm = a / (np.linalg.norm(a) + 1e-12)
    b_norm = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a_norm, b_norm) ** 2)

def _fidelity_adjacency(states: List[np.ndarray], threshold: float,
                        *, secondary: float | None = None,
                        secondary_weight: float = 0.5) -> nx.Graph:
    """Create a weighted graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for i, state_i in enumerate(states):
        for j, state_j in enumerate(states):
            if j <= i:
                continue
            fid = _state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
    return graph

# --------------------------------------------------------------------------- #
# Hybrid graph‑neural‑quantum layer
# --------------------------------------------------------------------------- #
class HybridGraphQLayer:
    """Hybrid graph‑neural‑quantum layer that fuses classical
    and quantum outputs into a weighted graph.

    Parameters
    ----------
    n_features : int
        Number of input features per node.
    n_nodes : int
        Number of nodes in the graph.
    n_qubits : int
        Number of qubits in the quantum circuit (only for interface).
    threshold : float, optional
        Fidelity threshold for graph edges.
    secondary : float | None, optional
        Secondary fidelity threshold for weighted edges.
    """
    def __init__(self,
                 n_features: int,
                 n_nodes: int,
                 n_qubits: int = 1,
                 threshold: float = 0.9,
                 secondary: float | None = None) -> None:
        self.n_features = n_features
        self.n_nodes = n_nodes
        self.n_qubits = n_qubits
        self.threshold = threshold
        self.secondary = secondary

        # classical linear layer
        self.classical = _ClassicFCL(n_features)

    def run(self,
            features: torch.Tensor,
            thetas: Iterable[float]) -> Tuple[np.ndarray, nx.Graph]:
        """
        Run the hybrid layer.

        Parameters
        ----------
        features : torch.Tensor
            Input features of shape (n_nodes, n_features).
        thetas : Iterable[float]
            Parameter values for the quantum circuit, one per node.

        Returns
        -------
        outputs : np.ndarray
            Combined classical + quantum output of shape (n_nodes,).
        graph : networkx.Graph
            Weighted graph built from quantum state fidelities.
        """
        # classical forward
        with torch.no_grad():
            class_outputs = self.classical(features).squeeze(-1).cpu().numpy()

        # quantum expectation simulation
        q_expect = _quantum_expectation(thetas)

        # combine (e.g., sum)
        combined = class_outputs + q_expect

        # build graph adjacency from quantum expectations
        graph = _fidelity_adjacency(q_expect, self.threshold,
                                    secondary=self.secondary)

        return combined, graph

    @staticmethod
    def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
        """Return the absolute squared overlap between two state vectors."""
        return _state_fidelity(a, b)

__all__ = ["HybridGraphQLayer"]
