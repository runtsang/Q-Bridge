"""Hybrid classical‑quantum graph layer (quantum implementation).

This module defines a single class, `HybridGraphQLayer`, that:
* implements a classical fully‑connected layer using NumPy,
* runs a parameterized quantum circuit implemented in Qiskit for each node,
* builds a weighted graph from the fidelities of the quantum states,
* exposes a `run` method returning the combined classical‑quantum output and the graph.
The class interface mirrors the classical version for consistency.

The quantum part uses Qiskit and can be run on any compatible backend.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
import networkx as nx
import qiskit
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import Parameter
from qiskit.providers import BaseBackend

# --------------------------------------------------------------------------- #
# Classical component – fully‑connected layer
# --------------------------------------------------------------------------- #
class _ClassicFCL:
    """Simple linear layer with a tanh activation (NumPy implementation)."""
    def __init__(self, n_features: int = 1) -> None:
        self.n_features = n_features
        self.weights = np.random.randn(n_features, 1)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Return the tanh of the linear output."""
        z = x @ self.weights
        return np.tanh(z).squeeze(-1)

# --------------------------------------------------------------------------- #
# Quantum component – Qiskit circuit
# --------------------------------------------------------------------------- #
class _QuantumCircuit:
    """Parameterized quantum circuit using Qiskit."""
    def __init__(self,
                 n_qubits: int,
                 backend: BaseBackend | None = None,
                 shots: int = 100):
        self.n_qubits = n_qubits
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.theta = Parameter("theta")
        self.circuit = QuantumCircuit(n_qubits)
        self.circuit.h(range(n_qubits))
        self.circuit.barrier()
        self.circuit.ry(self.theta, range(n_qubits))
        self.circuit.measure_all()

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Execute the circuit for each theta and return the expectation value."""
        job = execute(self.circuit,
                      backend=self.backend,
                      shots=self.shots,
                      parameter_binds=[{self.theta: theta} for theta in thetas])
        result = job.result()
        counts = result.get_counts(self.circuit)
        probs = np.array(list(counts.values())) / self.shots
        states = np.array([int(s, 2) for s in counts.keys()])
        expectation = np.sum(states * probs)
        return np.array([expectation])

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
        Number of qubits in the quantum circuit.
    threshold : float, optional
        Fidelity threshold for graph edges.
    secondary : float | None, optional
        Secondary fidelity threshold for weighted edges.
    backend : qiskit.providers.BaseBackend | None, optional
        Quantum backend to use.
    shots : int, optional
        Number of shots for measurement.
    """
    def __init__(self,
                 n_features: int,
                 n_nodes: int,
                 n_qubits: int = 1,
                 threshold: float = 0.9,
                 secondary: float | None = None,
                 backend: BaseBackend | None = None,
                 shots: int = 100) -> None:
        self.n_features = n_features
        self.n_nodes = n_nodes
        self.n_qubits = n_qubits
        self.threshold = threshold
        self.secondary = secondary

        self.classical = _ClassicFCL(n_features)
        self.quantum = _QuantumCircuit(n_qubits, backend=backend, shots=shots)

    def run(self,
            features: np.ndarray,
            thetas: Iterable[float]) -> Tuple[np.ndarray, nx.Graph]:
        """
        Run the hybrid layer.

        Parameters
        ----------
        features : np.ndarray
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
        class_outputs = self.classical.forward(features)

        # quantum expectation
        q_expect = self.quantum.run(thetas).squeeze(-1)  # shape (n_nodes,)

        # combine
        combined = class_outputs + q_expect

        # graph from quantum expectations
        graph = _fidelity_adjacency(q_expect, self.threshold,
                                    secondary=self.secondary)

        return combined, graph

    @staticmethod
    def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
        """Return the absolute squared overlap between two state vectors."""
        return _state_fidelity(a, b)

__all__ = ["HybridGraphQLayer"]
